struct STFT{T,D} 
  data::Array{T,D}
  freq::Any
  time::Any
end

function stft_(data::Array{T,1}, n, noverlap, sampFreq::Float64;
               nfft::Int=nextfastfft(n)) where T<:Real
  gauss(M) = gaussian(M, 0.15)
  normFactor = sum(gauss(n)) / 2                                           
  sp = DSP.stft(data, n, noverlap; fs=sampFreq, window=gauss) / normFactor
  freq = rfftfreq(nfft, sampFreq)
  time = (n/2 : n-noverlap : (size(sp,2)-1)*(n-noverlap)+n/2) / sampFreq    
  return STFT(sp, freq, time)
end

function stft_(data::Array{T,D}, args...) where {T<:Real, D}
  # calculate the STFT for each channel
  stfts = [stft_(data[:,i], args...) for i in CartesianIndices(size(data)[2:end])]    
  data_ = similar(stfts[1].data, size(stfts[1].data)..., size(data)[2:end]...)  
  for i in CartesianIndices(size(data)[2:end]) 
    data_[:,:,i] = stfts[i].data
  end
  return STFT(data_, stfts[1].freq, stfts[1].time)  # reduce redundancy of freq and time vectors
end

function spectrogramAnalysis(sp::STFT, snrTreshold::Float64 = 0.0)
  mmrfreq = zeros(size(sp.data,2))
  Sv = zeros(size(sp.data)[2:end]...)
  Su = zeros(size(sp.data)[2:end]...)
  idxMax = argmax(abs.(sp.data[:,1,:]))[2]      

  # get instantaneous MMR frequency 
  for j in 1:size(sp.data,2)                                  
    mmrfreq[j] = mmrFrequency(sp.data[:,j,idxMax], sp.freq) 
  end

  # get start phase from first snippet
  dataFTV, dataFTU, frV, frU = splitSpectrum(sp.data[:,1,:], sp.freq, mmrfreq[1]*1.5)
  phase = mmrPhase(dataFTV, frV, mmrfreq[1])

  # define signs for amplitude based on the phase of the signal
  signsV = zeros(size(sp.data,3))
  signsU = zeros(size(sp.data,3))
  for c=1:size(sp.data,3)              
    phaseV_ = mmrPhase(dataFTV[:,c], frV, mmrfreq[1])     
    signsV[c] = (sqrt(2-2*cos(phase - phaseV_)) < 0.3) ? 1 : -1              
    phaseU_ = mmrPhase(dataFTU[:,c], frU, 2*mmrfreq[1])  
    signsU[c] = (sqrt(2-2*cos(mod2pi(2*phase) - phaseU_)) < 0.3) ? 1 : -1    
  end

  dataNoiseV, frNoise = selectSpectrum(sp.data[:,1,:], sp.freq[:], mmrfreq[1]*1.2, mmrfreq[1]*1.4)
  noiseV = std(dataNoiseV)
  dataNoiseU, frNoise = selectSpectrum(sp.data[:,1,:], sp.freq[:], mmrfreq[1]*2.2, mmrfreq[1]*2.4)
  noiseU = std(dataNoiseU)

  # get instantaneous amplitudes for each channel 
  for i in CartesianIndices(Sv)   
    dataFTV, dataFTU, frV, frU = splitSpectrum(sp.data[:,i], sp.freq[:], mmrfreq[i[1]]*1.5) 
    ampU = mmrAmplitude(dataFTU, frU)
    ampV = mmrAmplitude(dataFTV, frV)
    # signal less than snrTreshold, set to zero
    Su[i] = abs(ampU)/noiseU > snrTreshold ? signsU[i[2]]*ampU : 0.0     
    Sv[i] = abs(ampV)/noiseV > snrTreshold ? signsV[i[2]]*ampV : 0.0
  end

  for c=1:size(sp.data,3)   
    idx = findfirst(x -> x == 0, Su[:,c])
    if idx != nothing
      Su[idx:end,c] .= 0.0
    end
    idx = findfirst(x -> x == 0, Sv[:,c])
    if idx != nothing
      Sv[idx:end,c] .= 0.0
    end
  end

  return mmrfreq, Sv, Su, phase
end

function splitSpectrum(dataFT::Array{T,1}, fr::AbstractVector, freq::Float64) where {T<:Complex}
  idx = argmin(abs.(fr .- freq))
  return dataFT[1:idx], dataFT[idx+1:end], fr[1:idx], fr[idx+1:end]
end

function splitSpectrum(dataFT::Array{T,2}, fr::AbstractVector, freq::Float64) where {T<:Complex}
  idx = argmin(abs.(fr .- freq))
  return dataFT[1:idx,:], dataFT[idx+1:end,:], fr[1:idx], fr[idx+1:end]
end

function selectSpectrum(dataFT::Array{T,2}, fr::AbstractVector, minFreq::Float64, maxFreq::Float64) where {T<:Complex}
  idxMin = argmin(abs.(fr .- minFreq))
  idxMax = argmin(abs.(fr .- maxFreq))
  return dataFT[idxMin:idxMax,:], fr[idxMin:idxMax]
end

function spectrum(data::AbstractArray{T,D}, samplingFrequency::Float64) where {T<:Real,D}
  numSamples = size(data,1)
  win = gaussian(numSamples, 0.15)
  dataFT = rfft(win.*data,1) / (numSamples/2)
  fr = samplingFrequency*rfftfreq(numSamples)
  return dataFT, fr
end 

function localSpectrum(data::AbstractArray{T,D}, samplingFrequency::Float64) where {T<:Real,D}
  dataFT, fr = spectrum(data, samplingFrequency)
  I = CartesianIndices(size(data)[2:end])
  dataFT = dataFT[:,I]
  return dataFT, fr
end

function mmrAmplitude(dataFT::AbstractArray{T,D}, fr::AbstractVector) where {D,T<:Complex}
  dataFTAbs = abs.(dataFT)
  idx = argmax(dataFTAbs)
  idxp = CartesianIndex(idx[1]+1, ntuple(i->idx[i+1],length(idx)-1)...)
  idxm = CartesianIndex(idx[1]-1, ntuple(i->idx[i+1],length(idx)-1)...)
  I = ntuple(i->(1:size(dataFT,i+1)),length(idx)-1)

  if idx[1] > 1 && idx[1] < length(fr)
    Δ = log(dataFTAbs[idxp] / dataFTAbs[idxm]) / (2*log(dataFTAbs[idx]^2 / (dataFTAbs[idxp]*dataFTAbs[idxm])))
    return exp.( log.(dataFTAbs[idx[1],I...]) .+ 1/4*(log.(dataFTAbs[idxp[1],I...]) .- log.(dataFTAbs[idxm[1],I...]))*Δ ) 
  else
    return dataFTAbs[idx[1], I...]
  end
end


function mmrPhase(dataFT::Array{T,D}, fr::AbstractVector, mmrFreq::Float64) where {T<:Complex,D}    
  dataFTAbs = abs.(dataFT)
  idx = argmax(dataFTAbs)
  idx_ = idx[1]
  
  idxp = CartesianIndex(idx_+1, ntuple(i->idx[i+1],length(idx)-1)...)
  idxm = CartesianIndex(idx_-1, ntuple(i->idx[i+1],length(idx)-1)...)
  if idx_ > 1 && idx_ < length(fr)
    Δ = log(dataFTAbs[idxp] / dataFTAbs[idxm])/(2*log(dataFTAbs[idx]^2 / (dataFTAbs[idxp]*dataFTAbs[idxm]))) # gaussian

    if Δ < 0
      α = -Δ
      z = angle(dataFT[idxm]) > angle(dataFT[idx]) ? 0.0 : 2π # avoid phase wraps
      φ = α*(angle(dataFT[idxm])+z) + (1-α)*angle(dataFT[idx]) + π/2
    else
      α = 1 - Δ
      z = angle(dataFT[idx]) > angle(dataFT[idxp]) ? 0.0 : -2π # avoid phase wraps
      φ = α*angle(dataFT[idx]) + (1-α)*(angle(dataFT[idxp])+z) + π/2
    end   
    return mod2pi(φ)
  else
    return 0.0
  end
end

function mmrFrequency(dataFT::AbstractArray{T,D}, fr::AbstractVector) where {D,T<:Complex}
  dataFTAbs = abs.(dataFT)
  idx = argmax(dataFTAbs)

  idxp = CartesianIndex(idx[1]+1, ntuple(i->idx[i+1],length(idx)-1)...)
  idxm = CartesianIndex(idx[1]-1, ntuple(i->idx[i+1],length(idx)-1)...)
  if idx[1] > 1 && idx[1] < length(fr)
    Δ = log(dataFTAbs[idxp] / dataFTAbs[idxm]) / (2*log(dataFTAbs[idx]^2 / (dataFTAbs[idxp]*dataFTAbs[idxm])))
    return fr[idx[1]]+Δ*(fr[2] - fr[1])
  else
    @info "Careful, no gauss interpolation for frequency"
    return fr[idx[1]]
  end
end

function doSTFT(data, windowLength, windowOverlap, samplingFrequency, thresholdSNRfreq)
  windowLength = collect(windowLength)
  STFTGrid = Matrix{Any}(undef, length(windowLength), 1)
  phaseGrid = zeros(length(windowLength), 1)
  for (indPeriods,windowLength) in enumerate(windowLength)
      stfts = stft_(data, windowLength, windowOverlap, samplingFrequency)
      times = cutTimeSamples/samplingFrequency .+ (stfts.time)
      freq, Av, Au, phase = spectrogramAnalysis(stfts, thresholdSNRfreq)

      numPhaseSamples = round(Int,5*samplingFrequency/freq[1])
      dataFTPhase, frPhase = localSpectrum(data[1:numPhaseSamples,:], samplingFrequency)
      phase = mod2pi(mmrPhase(dataFTPhase, frPhase, freq[1]) + π) 

      STFTGrid[indPeriods, 1] = (collect(times), freq, Av, Au)
      phaseGrid[indPeriods, 1] = phase
  end

  # combine all STFTs and sort by time
  all_Data = [collect(zip(e[1], e[2], zip(eachcol(e[3])...), zip(eachcol(e[4])...))) for e in STFTGrid] 
  combinedSTFT = vcat(all_Data...)  
  sortedSTFT = sort(combinedSTFT, by = x -> x[1])

  timeAll = [e[1] for e in sortedSTFT]
  freq = [e[2] for e in sortedSTFT]
  
  # Extract amplitudes and determine signs based on mean
  Av = mapreduce(a -> sign(mean([e[3][a] for e in sortedSTFT])) .* abs.([e[3][a] for e in sortedSTFT]), hcat, 1:size(data,2))
  Au = mapreduce(a -> sign(mean([e[4][a] for e in sortedSTFT])) .* abs.([e[4][a] for e in sortedSTFT]), hcat, 1:size(data,2))

  phase = mean(phaseGrid)

  # set values to zero after signal drops below threshold
  for c=1:size(Av,2)   
    idx = findfirst(x -> x == 0, Av[:,c])
    if idx != nothing
      Av[idx:end,c] .= 0.0
    end
    idx = findfirst(x -> x == 0, Au[:,c])
    if idx != nothing
      Au[idx:end,c] .= 0.0
    end
  end

  return timeAll, freq, Av, Au, phase
end
