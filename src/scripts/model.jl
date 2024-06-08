using Distributions
using GaussianKDEs
using GaussianMixtures
using LinearAlgebra
using StatsBase

const sigma_x = 0.01
const sigma_y = 0.1
const sigma_z = 0.4
const sigma_w = 0.1

function observe(dphi, m1, q, chi_eff)
    m2 = q*m1
    mt = m1+m2
    eta = m1*m2/(mt*mt)

    mc = mt*eta^(3/5)
    log_mc = log(mc)

    logit_q = log(q) - log1p(-q)

    x = rand(Normal(log_mc + dphi, sigma_x))
    y = rand(Normal(log_mc, sigma_y))
    z = rand(Normal(logit_q, sigma_z))
    w = rand(Normal(q * (one(chi_eff) + chi_eff), sigma_w))

    (x,y,z,w)
end

function sample_posterior(x, y, z, w; n=16384)
    xs = rand(Normal(x, sigma_x), n)
    ys = rand(Normal(y, sigma_y), n)
    zs = rand(Normal(z, sigma_z), n)
    ws = rand(Normal(w, sigma_w), n)

    qs = one(z) ./ (one(z) .+ exp.(.-zs))
    log_mcs = ys
    dphis = xs .- log_mcs
    chi_effs = ws ./ qs .- one(z)

    mcs = exp.(log_mcs)
    etas = qs ./ (one(z) .+ qs).^2
    mts = mcs ./ etas.^(3/5)
    m1s = mts ./ (one(z) .+ qs)

    dydmcs = one(y) ./ mcs
    dmcdm1s = (one(y) .+ qs) .* etas.^(3/5)
    dzdqs = one(z) ./ qs + one(z) ./ (one(z) .- qs)
    dwdchie = qs

    return ((dphis, m1s, qs, chi_effs), dydmcs .* dmcdm1s .* dzdqs .* dwdchie)
end

function resample_to_likelihood(dp, m1, q, chi_eff, prior_wts)
    w = 1 ./ prior_wts
    rs = rand(Uniform(0, maximum(w)), length(w))

    sel = rs .< w

    return (dp[sel], m1[sel], q[sel], chi_eff[sel])
end

function fit_gmm(dpsl, chi_effsl, m1sl, qsl; n=3)
    GMM(n, hcat(dpsl, chi_effsl, m1sl, qsl), kind=:full)
end

const alpha = 2.35
const beta = 2.0
const mu_eff = 0.1
const sigma_eff = 0.1

function draw_population()
    # m ~ m^-alpha
    m = rand(Pareto(alpha-1))
    # q ~ q^beta => Q ~ Q^-beta d(q)/d(Q) = Q^(-beta - 2) => Q ~ Pareto(beta+1)
    qinv = rand(Pareto(beta+1))
    q = one(qinv) / qinv

    dphi = 0.0

    chi_eff = rand(Normal(mu_eff, sigma_eff))

    (dphi, m, q, chi_eff)
end

function refactor_gaussians(x, C, mu, Lambda)
    nd = size(x,1)
    ng = size(mu,1)

    Linv = inv(Lambda)
    Cinv = inv(C)

    Ainv = Cinv
    Ainv[1:ng,1:ng] += Linv

    A = inv(Ainv)

    Linv_mu = zeros(nd)
    Linv_mu[1:ng] = Linv * mu

    a = A * (Linv_mu + Cinv * x)

    a = a[ng+1:end]
    A = A[ng+1:end, ng+1:end]

    b = mu
    B = C[1:ng,1:ng] + Lambda

    return (a, A, b, B)
end

function translated_gmm(gmm, mu, Lambda)
    n = gmm.n
    ng = size(mu, 1)

    gaussians = []
    for i in 1:n
        C = inv(gmm.Σ[i]' * gmm.Σ[i])
        a, A, b, B = refactor_gaussians(gmm.μ[i,:], C, mu, Lambda)

        w = gmm.w[i] * pdf(MultivariateNormal(b, Hermitian(B)), gmm.μ[i,1:ng])

        push!(gaussians, (w, a, Hermitian(A)))
    end

    gaussians
end

"""
    translated_kde(kde, mu, Lambda)

Given a KDE representation of a likelihood function and the mean and covariance
of a Gaussian population model for a subset of the parameters, return `(log_wts,
samples)`, weights and samples over the other parameters representing the
likelihood marginalized over the Gaussian population.

The Gaussian components of the population are assumed to be parameters
`1:n_gaussian` out of `1:n` total parameters represented in the KDE.  The
returned log-weight array will have shape `(n_pts,)`, the same length as the
number of points in the KDE, and the samples returned will have shape
`(n-n_gaussian, n_pts)`.
"""
function translated_kde(kde, mu, Lambda)
    Lc = cholesky(Hermitian(Lambda))
    Cc = kde.chol_bw

    C = Cc.L * Cc.U

    ng = size(mu, 1)
    n = size(Cc, 1)

    Linv = Lc \ I
    Linv_full = cat(cat(Linv, zeros(n-ng, ng), dims=1), zeros(n, n-ng), dims=2)

    Lambda_full = cat(cat(Lambda, zeros(n-ng, ng), dims=1), zeros(n, n-ng), dims=2)

    Cinv = Cc \ I

    Ainv = Linv_full + Cinv
    Ainvc = cholesky(Hermitian(Ainv))

    B = Lambda_full + C
    B = B[1:ng,1:ng]
    b = mu

    marg_dist = MultivariateNormal(b, B)

    mu_pred = cat(Lc \ mu, zeros(n-ng), dims=1)

    log_wts = [logpdf(marg_dist, kde.pts[1:ng,j]) for j in axes(kde.pts, 2)]
    pts = [(Ainvc \ (mu_pred + Cc \ kde.pts[:,j]))[ng+1:end] for j in axes(kde.pts,2)]

    return (log_wts, [pts[j][i] for i in 1:n-ng, j in axes(kde.pts,2)])
end

dp, m1, q, chi_eff = draw_population()
x, y, z, w = observe(dp, m1, q, chi_eff)
((dps, m1s, qs, chi_effs), prior_wts) = sample_posterior(x, y, z, w)
dpsl, m1sl, qsl, chi_effsl = resample_to_likelihood(dps, m1s, qs, chi_effs, prior_wts)

mu = [0.0, mu_eff]
Lambda = diagm([0.01*0.01, sigma_eff*sigma_eff])
gaussian_pop = rand(MultivariateNormal(mu, Lambda), 1024)

k = KDE(vcat(dpsl', chi_effsl', m1sl', qsl'))
log_wts, pts = translated_kde(k, mu, Lambda)
wts = exp.(log_wts .- maximum(log_wts))
wts = wts ./ sum(wts)
inds = sample(1:length(wts), Weights(wts), 1024)
samples = pts[:,inds]

c1 = Makie.wong_colors(0.5)[1]
c2 = Makie.wong_colors(0.5)[2]
c3 = Makie.wong_colors(0.5)[3]
pairplot(PairPlots.Series((; dphi=dpsl, chi_eff=chi_effsl, m1=m1sl, q=qsl), label="Likelihood", color=c1, strokecolor=c1), 
         PairPlots.Series((; dphi=gaussian_pop[1,:], chi_eff=gaussian_pop[2,:]), label="Gaussian Population", color=c2, strokecolor=c2), 
         PairPlots.Series((; m1=samples[1,:], q=samples[2,:]), label="Marginal Likelihood", color=c3, strokecolor=c3),
         PairPlots.Truth((; dphi=dp, chi_eff=chi_eff, m1=m1, q=q), label="Truth", color=:black, strokecolor=:black);
         labels = Dict(:dphi => L"\delta \phi", 
                       :chi_eff => L"\chi_{\mathrm{eff}}",
                       :m1 => L"m_1",
                       :q => L"q"))