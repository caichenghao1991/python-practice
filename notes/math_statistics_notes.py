"""
    statistics
    Uncertainty: caused by imperfect data(only fraction  of / indirect sign/ noisy data is observable)
    probability:
        experiment: result contain uncertainty
        sample space: all outcome of experiment
        event: subset of sample space
        random variable (X): enumerate sample space to real value
        category: discrete, continuous
        probability of random variable:
            probability density function(PDF): F(x) ~ P(X=x), For X frequency of possible outcome x's occurrence of the
                total possible occurrence. (function of probability of each possible outcome, sum probability is 1)
            cumulative density function(CDF): F(x) ~ P(X<=x),  probability that X will take a value less than or equal
                to x.    lim [x->-∞] F(x) = 0, lim [x->∞] F(x) = 1
            PDF -> CDF: is taking integral.    CDF -> PDF: is taking derivative.
            can use graph xϵ[0,1], yϵ[0,1] square with (normalized) length 1 represent sample space for uniform
                distribution, then desired possibilities (some relation between x and y represent as slicing edge inside
                the square) can be present as the area between slicing edges inside the square
            conditional probability: probability of A given B: P(B|A) = P(A ∩ B)/P(A)
                independence: A and B independent: P(A ∩ B) = P(A) * P(B)    or P(A|B) = P(A)
                    A and B conditional independent: P(AB|C) = P(A|C) * P(B|C)
                conditional independent can't infer independent, and independent can't infer conditional independent
            Bayes theorem: P(B|A) = P(A|B) * P(B) / P(A) = P(A|B) * P(B) /(P(A|B)*P(B) + P(A|~B)*P(~B))  ~B: not B
                P(B) called prior probability and P(B|A) called Posteriori probability
        expectation: measure the center of a random variable. discrete: E(X) = Σ [x] (x*P(x))
            continuous: E(X) = ∫ [x] (x*P(X=x))    E(k*X) = k*E(X)
            population mean vs sample mean: sample mean is a random variable, estimator of population mean. if unbiased
                E(sample mean) = population mean.
        Variance (σ^2): measure how random variable spread.  Standard deviation: σ, square root of variance
            Var(X) = E[(X-μ)^2] = E[X^2] - E[X]^2       μ = E(x)
            Var(X+Y) = Var(X) + Var(Y)  if X and Y independent
            large sample size better because lower sample variance
            sample variance (s^2):  s^2 = Σ[i] (((X_i)-X_mean)^2 / (n-1))      X_mean: average for n samples
                (n-1 make s^2 unbiased, same expectation)

        Bernoulli distribution: P(X=x) = p^x * (1-p)^1-x   (x=1 or 0)      E(X) = p   Var(X) = p(1-p)
            Binomial: sum of N Bernoulli variable:  P(X=x) = C_Nx p^x(1-p)^(N-x)     C_Nx: x times in total N times
               C_Nx = N! / x!(N-x)!   E(X) = N*p, Var(X) = N*p(1-p)
        Geometric distribution: probability of seeing first success with k independent trials, each with success
            probability p:  Pr(X=k) = (1-p)^(k-1) * p   E(X) =1/p
        Negative binomial distribution: probability of seeing r failures and k success trials,(last one failure) each
            with success probability p:  Pr(X=k) = C_(k+r-1)k p^k * (1-p)^r       C_(k+r-1)k =(k+r-1)! / k!(r-1)!


        Poisson: random variable count X can only be integer >= 0, over a period of time. count data per time unit
            relevant to binomial if slicing time into infinite splits. distribution x axis is k, y axis P(X=k)
            possibility of count=k in a time unit given average counts per time unit
            P(X=k) =  (λ^k /k!) * e^(-λ)   λ: counts per time unit,  k: specific value k

        Exponential distribution: in a poisson distribution, inter-arrival time(time between count increase)(possibility
            of increase count by 1 over t units of time passed) follows exponential distribution
            distribution x axis is t, y axis P(T=t)     random variable T is unit of time passed
            PDF: f(t) = P(T=t) = λe^(-λt)   CDF: P(T<=t) = 1 - e^(-λt) (possibility of count not increase in t unit of
                time)  E[t] = 1/λ   Var[t] = 1 / λ^2
            memory-less: P(T>s+t|T>s) = P(T>t)    # inter-arrival time don't depend on entry point of time (when start
                counting)

        Normal distribution:
            P(X=x) = 1/(σ*sqrt(2π)) * e^(-0.5*((x-μ)/σ)^2)    E[X] = μ   Var[X] = σ^2    random variable is value of x
                X = μ + σZ ~ N(μ,σ^2)  distribution x axis is value of x, y axis is P(X=x)
            standard normal distribution: μ=0, σ=1    P(X=x) = 1/(sqrt(2π)) * e^(-0.5 * x^2)
                [-σ,σ] cover 68%, [-2σ,2σ] cover 95%, [-3σ,3σ] cover 99.7%

        Chi-squared distribution (X^2) with k degrees of freedom is the distribution of a sum of the squares of k
            independent standard normal random variables.  distribution x axis is value of x, y axis is P(X=x)
        Log Normal distribution:  log(X) ~ N(μ,σ^2)   log of random variable X follow normal distribution
        Inverse transform sampling: Generate some distribution with normal distribution: X follows xx distribution,
            calculate CDF(must have cdf) of X: F(X) and inverse (from cumulative distribution [0,1] -> x value) F^-1(X),
            generate U with uniform distribution u, calculate F^-1(u), repeat n times.
        Acceptance rejection sampling: if meet the requirement keep the data, else drop. keep data with probability from
            PDF function. for generate a value v, PDF(v) = p, generate a uniform distribution u [0,1], if u<p keep it,
            otherwise reject it

        Percentile: sort data, if n% smaller than the data, then it's n% percentile. median(50%), 1st quantile(25%), 3rd
            quantile(75%)
            Z score: area outside  [-z, z] is α (alpha), Z_[α/2] = -1.96 for α=0.05




















"""