#################################################
# Parameter Estimation for Model-Based Sensing of
# Magneto-Mechanical Resonators			#
#################################################
# Julia: 1.11

using Pkg
Pkg.activate(".")
Pkg.instantiate()

using LsqFit
using OrdinaryDiffEq
using HDF5
using DSP
using FFTW
using Statistics
using SpecialFunctions
using Interpolations
using GLMakie

include("utils/utils.jl")
include("utils/method_T_T.jl")
include("utils/time_frequency_analysis.jl")

####################
# general settings #
####################
MMR_S = false     # whether to use MMRS or MMRL measurement
dataDir = "data/Experiment_2"   # directory where data is stored, needs to be adapted depending on whethere Experiment_1 or Experiment_2 data should be used
frameNo = 40          # frame on which to do single frame plots
tRX1 = 0.5            # percentage of RX1 size to use for fitting (e.g. 0.5 means we only use the first 50% of RX1 data for fitting)
if MMR_S
    @info "Using MMR S"
    dataName = "MMRS.h5"
    numPeriodsPerWindow = collect(14:1:17)
    thresholdSNRfreq = 1.5
    banddefs = [600.0,2000.0]       # passband frequencies for bandpass filter (in Hz)
    excitationFreq = 811.0
    remanentMagnetization = 1.44e6  # remanent magnetization of the rotor magnet (in uT)
    radius = 0.5                 # radius of the rotor magnet (in mm)
else
    @info "Using MMR L"
    dataName = "MMRL.h5"
    numPeriodsPerWindow = collect(9:1:11)
    thresholdSNRfreq = 1.0
    # setting for BandpassFilter
    banddefs = [80.0,300.0]         # passband frequencies for bandpass filter (in Hz)  
    order = 4
    excitationFreq = 111.0
    remanentMagnetization = 1.27e6  # remanent magnetization of the rotor magnet (in uT)
    radius = 2                  # radius of the rotor magnet (in mm)
end
order = 4                           # order of Butterworth filter

#############
# Load Data #
#############
@info "Loading data..."
data, fs = readDataFromHDF5(joinpath(dataDir, dataName))

########################################
# settings parameter estimation method #  
########################################
windowOverlap = 0             # overlap of windows for STFT (in samples) 
minRelAmplitude = 0.3         # minimum relative amplitude to consider for fitting (e.g. 0.3 means we only use data until the signal has decreased to 30% of its maximum amplitude)
maxIterations = 100           # maximum number of iterations for optimization
windowLength = Vector{Int}(undef, length(numPeriodsPerWindow))    # window lengths for STFT (in number of samples), will be calculated based on the number of periods per window and the excitation frequency
for (i, numPeriods) in enumerate(numPeriodsPerWindow)
  windowLength[i] = round(Int, numPeriods*fs / excitationFreq)
end

########################
# post-processing data #
########################
cutTimeSamples =  settlingTime(banddefs, order, fs; threshold_ratio = 0.01)   # number of samples to cut off at the beginning and end of the signal after filtering
filteredData = apply(banddefs, order, data[:,:,frameNo], fs)[cutTimeSamples:end-cutTimeSamples,:]  # apply bandpass filter and cut off beginning and end of signal to remove filter artifacts

idx = trunc(Int32,maxIdxWithSignal(filteredData, fs, minRelAmplitude, n= maximum(windowLength)) * tRX1)

m_r = 4/3 * π * (radius*10^-3)^3 * remanentMagnetization  # magnitude of the magnetic moment vector of the rotor
filteredData = filteredData./m_r                  # normalize data to magnetic moment of rotor 
filteredData_ = filteredData[1:idx,:]        # reduce RX1 size based on tRX1
###################################################
# get initial parameters, constraints and weights #
###################################################
timeAll, freq, Av, Au, ψstart_simp = doSTFT(filteredData_, windowLength, windowOverlap, fs,thresholdSNRfreq) # do STFT for all window lengths and combine afterwards
# get initial parameters for optimization based on STFT results

ωnat_simp = maximum(freq)
φmax_simp = freqToφmax(minimum(freq), ωnat_simp)
damping_simp = dampingTerm(Av,timeAll)
σV_simp = -Av[1,:] / (-ωnat_simp*2*pi*pi^3 * kappaV(φmax_simp))
σU_simp = Au[1,:] / (ωnat_simp*2*pi*4*pi^3 * kappaU(φmax_simp)) 
startparams = vcat([φmax_simp, damping_simp, ωnat_simp, ψstart_simp], σV_simp, σU_simp)
# get constraints for optimization 
lb = vcat([0, 0.001, ωnat_simp/4, 0], fill(-Inf, 2*size(data,2)))   # lower bounds for φmax corresponds to 0 deg)
ub = vcat([π/2, Inf, ωnat_simp*4, 2*π], fill(Inf, 2*size(data,2))) # upper bounds for φmax 

# define weights
weights = ones(length(filteredData_)) ./ maximum(abs.(filteredData_))

# define model function for optimization
function model_(t, params)
  model = ODEModel(params[1], params[2], params[5:(5+size(filteredData_,2)-1)], 
  params[(5+size(filteredData_,2)):(5+2*size(filteredData_,2)-1)], params[3], params[4], extrema(t))
  return vec(recoverSignal(model, t))
end

########################
# perform optimization #
########################
@info "Estimating parameters..."
times_ = range(cutTimeSamples*1/fs, step=1/fs, length=size(filteredData_,1))  # time vector for fitting
fit = curve_fit(model_, times_, vec(filteredData_), weights .^2, startparams, maxIter=maxIterations, lower=lb, upper=ub; 
                    autodiff=:finiteforward)

φmax_ = fit.param[1]
D = fit.param[2]
fnat_ = fit.param[3]
phase = fit.param[4]
Sv = fit.param[5:(5+size(filteredData_,2)-1)]
Su = fit.param[(5+size(filteredData_,2)):(5+2*size(filteredData_,2)-1)]

# reconstruct fitted signal based on optimized parameters
times = range(cutTimeSamples*1/fs, step=1/fs, length=size(filteredData,1))  # time vector for fitting
dataFit = recoverSignal(ODEModel(φmax_, D, Sv, Su, fnat_, phase, extrema(times)), times)

############
# Plotting #
############
fig = Figure(size=(1400, 900))
ymax = maximum(abs.(filteredData)) * 1.1
ymin = -ymax
col = 1
for ch in 1:size(filteredData,2)
  ax = Axis(fig[ch, col:col+1], title="Channel $ch", xlabel="t / s", ylabel="s(t) / V")
  lines!(ax, times, filteredData[:,ch], label=L"s^{\mathrm{meas}}", color=colors[1])
  lines!(ax, times, dataFit[:,ch], label=L"s^{\mathrm{T}}", color=colors[2])
  vlines!(ax, times[idx], color=colors[3], linestyle=:dash, label=L"T_{\mathrm{RX1}}")
  xlims!(ax, times[1], times[end])
  ylims!(ax, ymin, ymax)
  if ch == 1
   fig[ch,3] =  Legend(fig, ax, framevisible = false, orientation = :vertical)
  end
end

col += 3
# plot STFT results
axSTFT = Axis(fig[1,col:col+1], title="Frequency", xlabel="t / s", ylabel="ω(t) / 2π x Hz")
scatter!(axSTFT, timeAll, freq)
xlims!(axSTFT, timeAll[1], timeAll[end])


  # print inital parameters in figure
  paramTextInit = "Initial parameters:\nφmax=$(round(rad2deg(φmax_simp), digits=3))°\nλ=$(round(damping_simp/2, digits=3)) s⁻¹\nfnat=$(round(ωnat_simp, digits=3)) Hz\nψstart=$(round(ψstart_simp, digits=3))\nσV=$(round.(σV_simp*10^3, sigdigits=3)) mT/A\nσU=$(round.(σU_simp*10^3, sigdigits=3)) mT/A"
  infoTextInit = Label(fig[2,col],
    justification = :left,
    lineheight = 1.1, font=:bold,
    tellwidth=false 
  )
infoTextInit.text = paramTextInit
# print fitted parameters in figure
paramText = "Fitted parameters:\nφmax=$(round(rad2deg(φmax_), digits=3))°\nλ=$(round(D/2, digits=3)) s⁻¹\nfnat=$(round(fnat_, digits=3)) Hz\nψstart=$(round(phase, digits=3))\nσV=$(round.(Sv*10^3, sigdigits=3)) mT/A\nσU=$(round.(Su*10^3, sigdigits=3)) mT/A"
infoText = Label(fig[2,col+1],
    justification = :left,
    lineheight = 1.1, font=:bold,
    tellwidth=false 
  )
infoText.text = paramText

rowsize!(fig.layout, 1, Fixed(200))
rowsize!(fig.layout, 2, Fixed(200))
rowsize!(fig.layout, 3, Fixed(200))




display(fig)

