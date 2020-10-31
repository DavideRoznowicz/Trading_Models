# Trading_Models
BSc Thesis & two backtesting models for StatArb portfolios: Pairs-Trading & Statistical Arbitrage

The backtesting simulations were the goal of my bachelor's thesis in Applied Mathematics, entitled "Development and implementation of a statistical arbitrage model for a US-equity-based portfolio". 
The first file implements a Pairs Trading model, while the second a more complex basket-based statistical arbitrage technique, using Johansen cointegration test among the securities of the clusters spotted by k-means algorithm. 
They've both been written in Python exploiting QuantConnect libraries and platform to lead the simulation in a proper way.

In the final implementation, execution costs have not been taken into account due to time restrictions; however a thorough description of optimal transaction costs management, based upon Robert Almgren and Neil Chriss work (Optimal execution of portfolio transactions), can be retrieved in my thesis (unfortunately only available in italian).
