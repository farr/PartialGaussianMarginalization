using Distributions
using GaussianMixtures

const sigma_x = 0.01
const sigma_y = 0.1
const sigma_z = 0.8

function observe(dphi, m1, q)
    m2 = q*m1
    mt = m1+m2
    eta = m1*m2/(mt*mt)

    mc = mt*eta^(3/5)
    log_mc = log(mc)

    logit_q = log(q) - log1p(-q)

    x = rand(Normal(log_mc + dphi, sigma_x))
    y = rand(Normal(log_mc, sigma_y))
    z = rand(Normal(logit_q, sigma_z))

    (x,y,z)
end

function sample_posterior(x, y, z; n=16384)
    xs = rand(Normal(x, sigma_x), n)
    ys = rand(Normal(y, sigma_y), n)
    zs = rand(Normal(z, sigma_z), n)

    qs = one(z) ./ (one(z) .+ exp.(.-zs))
    log_mcs = ys
    dphis = xs .- log_mcs

    mcs = exp.(log_mcs)
    etas = qs ./ (one(z) .+ qs).^2
    mts = mcs ./ etas.^(3/5)
    m1s = mts ./ (one(z) .+ qs)

    dydmcs = one(y) ./ mcs
    dmcdm1s = qs .* etas.^(3/5)
    dzdqs = one(z) ./ qs + one(z) ./ (one(z) .- qs)

    return ((dphis, m1s, qs), dydmcs .* dmcdm1s .* dzdqs)
end

function resample_to_likelihood(dp, m1, q, prior_wts)
    w = 1 ./ prior_wts
    rs = rand(Uniform(0, maximum(w)), length(w))

    sel = rs .< w

    return (dp[sel], m1[sel], q[sel])
end

function fit_gmm(dpsl, m1sl, qsl; n=4)
    GMM(n, hcat(dpsl, m1sl, qsl), kind=:full)
end


dp, m1, q = 0.0, 10.0, 0.8
x, y, z = observe(dp, m1, q)
((dps, m1s, qs), prior_wts) = sample_posterior(x, y, z)
dpsl, m1sl, qsl = resample_to_likelihood(dps, m1s, qs, prior_wts)
pairplot((; dphi=dpsl, m1=m1sl, q=qsl), PairPlots.Truth((; dphi=dp, m1=m1, q=q)))
gmm = fit_gmm(dpsl, m1sl, qsl)