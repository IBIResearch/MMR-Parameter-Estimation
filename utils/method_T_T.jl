mutable struct ODEModel 
  φmax::Float64
  D::Float64
  Sv::Vector{Float64}
  Su::Vector{Float64}
  fnat::Float64
  phase::Float64
  timeInterval::Tuple{Float64, Float64}
end

# Define the problem
function recoverDeflectionAngle_ode(du, u, p, t)
    θ = u[1]
    dθ = u[2]
    du[1] = dθ
    du[2] =  -p[1]^2 * sin(θ) - p[2] * dθ 
end

function recoverSignal(model::ODEModel, times)
   step_ = step(times)
  times_ = range(times[1], step=step_, length=length(times))

  ω0 = 2*π*model.fnat
  φmax = model.φmax
  D = model.D
  phase = model.phase

  tspan = (times_[1], times_[end])

  φdotmax = sqrt(2) * ω0 * sqrt(1 - cos(φmax))  # maximum angular velocity for a given maximum angle (derived through energy conservation)
  φ0 = cos(phase) * φmax
  φdot0 = -sin(phase) * φdotmax

  u₀ = [φ0, φdot0]
  params = (ω0, D)
  # Pass to solvers
  prob = ODEProblem(recoverDeflectionAngle_ode, u₀, tspan, params)
  sol = solve(prob, Tsit5(), abstol=1e-8,reltol=1e-8) 
  phi =  [sol(t_)[1] for t_ in times_]
  phiDiff = [sol(t_)[2] for t_ in times_]

  θv = phiDiff .* cos.(phi) 
  θu = - phiDiff .* sin.(phi) 

  U = θv*transpose(model.Sv) + θu*transpose(model.Su)
  return U
end