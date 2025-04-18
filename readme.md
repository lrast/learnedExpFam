# Learning Exponential family distributions by learning cumulant generating function

The idea of this project is simple: by training an input convex neural network on the empirical cumulant generating function, we can get an easy to use representation of the empirical density. This allows for:
1. Detection of statistics changes, based on the large deviations rate function.
2. Extension to a family of distributions by 'exponential tilting'
3. Determination of Fisher information

Here, I explore these ideas.
