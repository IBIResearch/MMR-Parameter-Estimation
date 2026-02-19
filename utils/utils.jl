  colors = [RGBf.(0/255,73/255,146/255), # blue
          RGBf.(239/255,123/255,5/255),	# orange (dark)
          RGBf.(138/255,189/255,36/255),	# green
          RGBf.(178/255,34/255,41/255), # red
          RGBf.(170/255,156/255,143/255), # mocca
          RGBf.(87/255,87/255,86/255),	# black (text)
          RGBf.(255/255,223/255,0/255), # yellow
          RGBf.(104/255,195/255,205/255),# "TUHH"
          RGBf.(45/255,198/255,214/255), #  TUHH
          RGBf.(193/255,216/255,237/255)]

"function to calculate the maximum index of the signal based on the decay of the amplitude"
function maxIdxWithSignal(data::Matrix{T}, samplingFrequency::Float64, minRelAmplitude::Float64; n::Int= 0) where T<:Real
  sp = stft_(data, n, 0, samplingFrequency)
  amp = zeros(size(sp.data)[2])
  maxChan = argmax(abs.(sp.data[:,1,:]))[2]
  for i in CartesianIndices(amp)      # get instantaneous amplitude of the channel
    amp[i] = mmrAmplitude(vec(sp.data[:,i,maxChan]), sp.freq)
  end
  
  relAmp = amp ./ maximum(amp)
  idx = findfirst(x -> x < minRelAmplitude, relAmp)
  idx = (idx == nothing) ? length(relAmp) : idx
  y = log.(relAmp[1:idx])
  x = sp.time[1:idx]
  ybar = mean(y)
  xbar = mean(x)
  a = sum((x .- xbar) .* (y .- ybar)) / sum((x .- xbar) .^ 2)
  b = ybar - a * xbar

  logMinRelAmplitude = log(minRelAmplitude)
  tsol = (logMinRelAmplitude - b) / a
  tsol = max(0.0, min(tsol, size(data,1)/samplingFrequency))

  if isnan(tsol)
    tsol = size(data,1)/samplingFrequency
  end
  idx = floor(Int, tsol * samplingFrequency)
  return idx
end

"function to read data from HDF5 file and extract relevant information"
function readDataFromHDF5(filePath::String)
  return h5open(filePath, "r") do file
    data = read(file, "data")
    sequenceGroup = file["sequence"]
    baseFrequency = read(sequenceGroup, "baseFrequency")
    decimation = read(sequenceGroup, "decimation")
    #numFrames = read(sequenceGroup, "numFrames")
    samplingFrequency = baseFrequency / decimation
    return data, samplingFrequency
  end
end

"function to get digital Butterworth bandpass filter"
function getDigitalFilter(banddef, order,fs)
  b = [2.0/fs .* (banddef[1], banddef[2])]
  return digitalfilter(Bandpass(b[1][1], b[end][2]), Butterworth(order))
end

"function to apply digital filter to signals using filtfilt for zero-phase distortion"
function apply(banddef, order, signals::Array{T,2}, fs) where {T<:Real}
  filt = getDigitalFilter(banddef, order, fs)
  return filtfilt(filt, signals)              
end

"function to get settling time of bandpass filter based on the impulse response of a digital filter"
function settlingTime(banddef, order, fs:: Float64; threshold_ratio = 0.01)
  filt = getDigitalFilter(banddef, order, fs)
  samples = trunc(Int,0.5 * fs)             
  t_resp = (0:samples-1) .* 1/fs
  impResp = impresp(filt, samples)      
  abs_signal = abs.(impResp)
  max_val = maximum(abs_signal)
  threshold = threshold_ratio * max_val     
  for i in 1:length(impResp)                
      if all(abs_signal[i:end] .<= threshold)
        return i
      end
  end
end


frequencyUndampedModel(φmax, fnat) =  π / (2*ellipk(sin(φmax/2)^2)) * fnat  

"function with interpolant for frequency to φmax conversion based on the undamped model"
function freqToφmax(f, fnat)
    _interpolantFreqToφmax = Ref{Any}(nothing)
    φmax = reverse(collect(range(0, 90 / 180 * π, 1000)))     
    f_ = frequencyUndampedModel.(φmax, 1)                                          
    _interpolantFreqToφmax[] = extrapolate(interpolate((f_,), φmax, Gridded(Linear())), Flat())
  return _interpolantFreqToφmax[](f / fnat)         
end

"function to estimate damping term based on the amplitude decay of the v-signal"
function dampingTerm(Av,time)
  idxMax = argmax(abs.(Av[1,:])) 
  amp = abs.(Av[:,idxMax])    
  relAmp = amp ./ maximum(amp)    
  y = log.(relAmp)
  x = time
  ybar = mean(y)
  xbar = mean(x)
  a = sum((x .- xbar) .* (y .- ybar)) / sum((x .- xbar) .^ 2)
  b = ybar - a * xbar
  return -a * 2
end

ellipticNome(φmax) = exp(-π*(ellipk(1-sin(φmax/2)^2))/(ellipk(sin(φmax/2)^2)))
"function to calculate the kappa value for v for the amplitude based on the assumption of the undamped oscillation"
function kappaV(φmax)
  q = ellipticNome(φmax)
  return sqrt(q) / ((1+q) * (ellipk(sin(φmax/2)^2)^3))    
end

"function to calculate the kappa value for u for the amplitude based on the assumption of the undamped oscillation"
function kappaU(φmax)
  q = ellipticNome(φmax)
  return  q / ((1+q)^2 * (ellipk(sin(φmax/2)^2)^3))
end
