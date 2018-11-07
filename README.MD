# Convex Optimization Project

## Data folder
- Brain data are name as "(Subject No)_(Region of Interest)_(LH).npy"
For example,
"S1_PPA_LH.npy" means it is the brain data for subject 1, recording from PPA area, in the left hemisphere in the brain. 
The dimension of the brain data matrix is (5254, v) where 5254 is number of trials, and v is the number of dimension in this area of the brain. 
- Each trial is correponsding with a presentation of an image, which has an dominant object on it. The order of presentation is spcified in "image_category.p".
- Word embedding vector are stored in "pix2vec_200.model". 200 is the dimension of the vectors. 
You can load this file with Gensim package.
<pre><code>from gensim.models import KeyedVectors
wv_model = KeyedVectors.load("../outputs/pix2vec_200.model", mmap='r')
pix2vec = wv_model.vectors
wv_list = list(wv_model.vocab)
</code></pre>
In this case the pix2vec has a dimension of 3115 x 200. 3115 is the number of objects/images. wv_list is a list of words (length=3115), where each entry corresponds to a row in the pix2vec matrix. 







