# Learning Exponential family distributions by learning cumulant generating function

The idea of this project is simple: by training an input convex neural network on the empirical cumulant generating function, we can get an easy to use representation of the empirical density. This allows for:
1. Detection of statistics changes, based on the large deviations rate function.
2. Extension to a family of distributions by 'exponential tilting'
3. Determination of Fisher information

Here, I explore these ideas.


## #Notes to self:
Previous versions:

See also the folder `~/Documents/postGraduation/projects/learningExpFams`, which contains work that I did several years ago, as well as lab notebook titled 'Learning Cumulant Functions', which is recent.


The reason that I'm returning to this project again now is that I have a better understanding of the relationship between the base measure and partition function in natural exponential families: the partition function represents the Cumulant Generating function of the base measure itself. This raises the idea of learning this function directly from data, and using it for inference in a neural network context. Further 

