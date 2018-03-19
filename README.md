# Online Image classification

### Goal:
- build a classifier of online images into advertisements and non-advertisements. 

### Data
-The dataset is described here: https://archive.ics.uci.edu/ml/datasets/Internet+Advertisements


### Tree with Maximum Depth of 3
The decision rules can be summarized as follows:
1. Large images (at least 300.2px in width and 44.5 px in height and 5.149 aspect ratio) are predicted to
be ads.
2. Images that are very wide but not particularly tall tend not to be ads (perhaps these are more likely to
be banners?).
3. For images that are smaller than 300.2 px in width, we might pay special attention to specific text
features in the destination URL.
• If the destination url contains ‘com’ but not ‘home’ and ‘html’, then we would predict the image to be
an ad. If the destination contains ‘com’, ‘home’, and ‘html’, then we would not expect the image to be
an ad.
• If the destination url does not contain ‘com’ but the image file itself resides in a location containing the
word ‘ads’, then it is also likely to be an ad. Images whose destination url does not contain ‘com’ and
whose image path does not contain the word ‘ads’ is not likely to be an ad.
None of these splitting rules seem counter-intuitive.

### Tree with Maximum Depth of 5

It turns out that this tree produces the same exact error rate as the depth 3 tree, even though it is 2 layers
deeper. The top 3 levels of the two trees are identical, as expected.


### Bagging(50 trees)
The out-of-bag error rate for the 50 trees is 2.56%. In fact, we see that this is quite close to the actual
out-of-sample error rate on the test set: 2.95%. The difference is that the out-of-bag error rate is computed
using the training observations that were not used in each step of the bagging procedure. As such, it behaves
like a cross-validation error rate, whereas the error rate on the test set is truly evaluated on a dataset that
none of the trees were trained on.

### Variable importance plot
