# 580 Final Project: NLP Application in Education:
## Creating a Auto Grader for 8-12th grade ESL Students
#### Authors: Yudi Chen (yc1000); Xinyue Ji (xj82); Hanjing Shi (hs1063); Ke Shi (ks1837)

## Introduction of Github

## Methodology utilized
**Introduction and EDA**

The use of auto-feedback systems to evaluate students\' writing
abilities is becoming increasingly common in education. These systems
rely on algorithms to assess writing in terms of grammar, word use, and
other aspects. However, current auto-feedback systems need to be better
suited for evaluating the writing of English as a Second Language (ESL)
students. In this study, we aim to develop an effective model for
assessing the writing skills of ESL students in grades 8-12.

The dataset used in this study consists of two parts: a training set
with 3911 student essays and scores in six different areas (cohesion,
syntax, vocabulary, phraseology, grammar, and conventions), and a test
set with three student essays, two book passages, and three essays
written by our group members.

The accompanying distribution figure shows the length of essays written
by ESL students in grades 8-12. The distribution is slightly skewed to
the right, with a mean length of approximately 440 words. This suggests
that few of the 3911 essays are particularly long.

![](https://github.com/hs1063/580_final/blob/main/imgs/img/image001.png){width="6.5in" height="2.9583333333333335in"}

In this study, we have visualized the score distributions for six
different areas of writing ability. These distributions are generally
normal, indicating that most students have average levels of writing
ability. The distribution of total scores, which is the sum of scores in
all six areas, is bimodal, with peaks at around 16.5 and 20. The mean
total score is approximately 18.5, indicating that the average score in
each area falls between 2.75 and 3.33.

![](https://github.com/hs1063/580_final/blob/main/imgs/img/image002.png){width="6.5in" height="1.4583333333333333in"}

To better understand how scores are assigned to individual students, we
explored the diversity of word usage in each essay using the
nltk.pos_tag() function. This allowed us to divide all words in the
essay into categories such as nouns, verbs, adjectives, and adverbs. We
then used a correlation matrix to see whether there was a correlation
between the types of words used and the six scores. However, no
significant correlation was found, so these factors were not included in
the model architecting process.

![](https://github.com/hs1063/580_final/blob/main/imgs/img/image003.png){width="6.5in" height="2.986111111111111in"}

![](https://github.com/hs1063/580_final/blob/main/imgs/img/image004.png){width="6.5in" height="3.3055555555555554in"}

To accurately reflect the efficiency of our models, we randomly split
20% of our data as our validation set and calculated relative measures
due to the small amount of test data. Additionally, we added five
passages, including a paragraph from Harry Potter, a paragraph from To
Kill a Mockingbird, and three essays written by ESL students at
different levels of education (high school, college application, and
college) to determine whether our model could be expanded for more
generalized use. By predicting the scores for these passages, we aimed
to see whether the model could be used for higher education or essays
written by native English speakers.

![](https://github.com/hs1063/580_final/blob/main/imgs/img/image005.png){width="6.5in" height="1.8888888888888888in"}

To prevent biases, the input for this study was the raw student essays.
We used several natural language processing (NLP) methods, including
word2vec, n-gram, multi-Roberta, and TF-IDF. In the regression part, we
built ANN, SVR, and Conv1d models. After training the models, we used
the MCRMSE to measure their accuracy, with lower numbers indicating
better performance. Since we used four models, we took the average of
the final results to calculate the overall accuracy.

![](https://github.com/hs1063/580_final/blob/main/imgs/img/image006.png){width="6.5in" height="2.361111111111111in"}

To further improve the accuracy of our models, we used the k-fold
method, which divides the dataset into k smaller sets. We found that
using k = 5 produced the best results, as shown in the accompanying
graph. The k-fold method allows for better evaluation of the models by
ensuring that each data point is used for training and testing. This
reduces the likelihood of overfitting and increases the generalizability
of the model.

![](https://github.com/hs1063/580_final/blob/main/imgs/img/image007.png){width="5.031602143482065in"
height="3.4218755468066493in"}

**TF-IDF method for text vectorization with the SVM algorithm for
regression**

The first model we used combined the TF-IDF method for text
vectorization with the SVM algorithm for regression. TF-IDF is a common
method for representing text data numerically, which allows it to be fed
into algorithms. We used the SVM algorithm, a popular machine-learning
method. Specifically, we used the SVR (support vector regression)
variant of SVM for regression.

Several parameters can be adjusted in the SVR function, including the
regularization parameter \"c\", the \"epsilon\" value for the
epsilon-tube, and the \"gamma\" value for the kernel coefficient. We
used the default \"RBF\" kernel.

![](https://github.com/hs1063/580_final/blob/main/imgs/img/image008.png){width="6.5in" height="2.6527777777777777in"}

The score of the MCRMSE for this model was 0.58, which is not
particularly good. This suggests that the combination of TF-IDF and SVR
may not be the most effective for this dataset and task.

![](https://github.com/hs1063/580_final/blob/main/imgs/img/image009.png){width="6.5in" height="1.2777777777777777in"}

After evaluating the results of the TF-IDF with SVR model, we found that
the third student in the dataset had the best results. By comparing this
student\'s data with their essay text, we found that they used a high
number of verbs. This suggests that the use of verbs may have influenced
the TF-IDF score and contributed to the better prediction of this
student\'s writing ability. It is also possible that other factors, such
as the student\'s overall writing style or the specific vocabulary they
used, may have contributed to the model\'s performance. Further analysis
would be needed to confirm these observations.

![](https://github.com/hs1063/580_final/blob/main/imgs/img/image010.png){width="6.5in" height="3.1527777777777777in"}

After analyzing the results of the TF-IDF with the SVR model, we found a
weak correlation between the TF-IDF scores and the grammar scores of the
students. This suggests that the use of the TF-IDF method alone may not
be effective for predicting grammar ability. To further improve the
performance of our model, we tried using different combinations of NLP
methods and machine learning algorithms. For example, we experimented
with using the ANN (artificial neural network) algorithm for regression.
This resulted in an MCRMSE score of 0.51, which is better than the score
obtained with the SVR algorithm.

For the simple ANN, we used three layers. The first layer had 64 units
and used the SoftMax activation function, while the second layer had 32
units and used the same activation function. The output layer had six
units corresponding to the six scores we were predicting. The results of
the TF-IDF with a simple ANN model showed that the third student still
had the best results among the three students. However, the worst scores
for the first and third students shifted from grammar to syntax, while
the worst score for the second remained in grammar.

Overall, the results of the SVR and ANN models suggest that the TF-IDF
method may not be effective for predicting syntax and grammar ability.
We will need to explore other methods to increase the accuracy of our
model.

![](https://github.com/hs1063/580_final/blob/main/imgs/img/image011.png){width="6.5in" height="1.0555555555555556in"}

We also tried using the n-gram method with the Conv1d algorithm and the
multi-Roberta method with the SVR algorithm. For each combination, we
used the k-fold method with k = 5 to evaluate the model\'s performance
and selected the combination that yielded the best results. We then
fine-tuned the parameters of the selected model to improve its
performance even further. Finally, we averaged the results of the four
models to obtain the overall accuracy of the model.

**Conv1D model with bigrams**

With the Conv1D model, one of the decisions we had to make when using
this model was whether to use n-grams and, if so, what number of n-grams
to use. We initially did not use n-grams and obtained an accuracy of
0.52. We then tried using bi-grams, trigrams, and higher-order n-grams
to see if any of these would improve the model\'s accuracy. We found
that bi-grams produced the highest accuracy, while n-grams with n = 3,
4, and 5 had worse accuracy. One possible reason is that high school
students may have a low degree of correlation between many words. As
shown in the accompanying graph, the accuracy of the bi-gram model
reaches 0.5. This suggests that using bigrams may be a useful approach
for improving the accuracy of the Conv1D model.

In this model, conv1d layers are used for training. Four layers were
chosen for the deep stack, as the shallow stack was used in previous
models and it was found that four layers performed best after comparing
the results of using two, three, and four layers. Average pooling was
used in the middle of the two layers to improve the results. The model
also uses four dense layers, with each layer having half the size of the
previous one to increase accuracy.

Conv1d layers are a type of convolutional layer often used in natural
language processing and time series analysis. They apply a convolution
operation to the input along the time dimension, allowing the model to
learn local patterns in the data. By using a deep stack of conv1d
layers, the model can learn increasingly complex patterns in the data
and improve its performance.

Using average pooling in the middle of the two layers is another
technique to improve the model\'s performance. Pooling is a method of
down sampling the input to reduce its dimensionality and allow the model
to focus on the most important information. Average pooling calculates
the average value of each feature in the input, reducing its size while
retaining important information.

The use of four dense layers, with each layer having half the size of
the previous one is a way of increasing the model\'s capacity to learn
and improve its accuracy. Dense layers are fully connected layers in
which each neuron receives input from all the neurons in the previous
layer. By reducing the size of each layer, the model can learn more
complex patterns without overfitting to the training data.

![](https://github.com/hs1063/580_final/blob/main/imgs/img/image012.png){width="3.386619641294838in"
height="7.1528040244969375in"}

The results of the Conv1D with bi-gram model are similar to those of the
TF-IDF model in that the second student has the lowest scores for each
feature among all three students. However, the Conv1D with bi-gram model
provides a better prediction for the other features compared to the
TF-IDF model. This suggests that the Conv1D with bi-gram combination may
be more effective for this dataset and task than the TF-IDF with SVR
combination. Further analysis and experimentation would be needed to
confirm this.

![](https://github.com/hs1063/580_final/blob/main/imgs/img/image013.png){width="6.5in" height="1.2638888888888888in"}

**Multi-Robot Model with SVR Algorithm**

The multi-robot model is the third model that was tested in this study.
It is based on the BERT model and uses an autoencoder to train the
model. The input matrix for this model is very large, which caused the
kernel to crash when it was trained on Amazon AWS Cluster and Google
Colab.

To improve the model\'s accuracy, additional pretrained architectural
embeddings such as Robertabase and robertalarge were added to the
distilled roberta model. However, this caused the accuracy to decrease
to 0.58.

To address this issue, pooling was applied to the model to downsample
the input and reduce noise information. Mean pooling was used to
calculate the average token-level BERT embedding for each essay. In this
study, mean pooling was found to be more effective than max pooling and
mean square root length pooling because it allowed more information to
be retained without altering the content of the essay.

![](https://github.com/hs1063/580_final/blob/main/imgs/img/image014.png){width="6.5in" height="2.611111111111111in"}

To improve the precision of the multi-roberta model, Huber Loss was
selected as the loss function. Huber Loss combines the benefits of
L1Loss and MSELoss and is less sensitive to outliers in the data than
squared error loss. However, this caused the accuracy score to decrease
to 0.52.

An SVR layer was then added to the model in an attempt to boost its
accuracy. SVR, or Support Vector Regression, is a type of algorithm that
works well with high-dimensional data and clear margins of separation.
In this model, the regularization parameter for the SVR was set to 10,
which resulted in an accuracy score of 0.47, the lowest among all the
models.

Like the other three models, the multi-roberta model had lower scores
for the second student\'s phraseology and grammar compared to the other
students. This is likely due to the fact that the second student\'s
essay was the shortest and contained the fewest adjectives and adverbs.

![](https://github.com/hs1063/580_final/blob/main/imgs/img/image015.png){width="6.5in" height="1.1388888888888888in"}

**Conclusion**

The final result for the prediction was obtained by averaging the
results of all the models to adjust for potential biases introduced by
individual models. The MCRMSE for this result was 0.48, which was higher
than three of the models but still lower than the Multi-Roberta model.
According to the result, the best writer was the third student, followed
by the first student, and the worst writer was the second student.

To further improve the model, we could consider assigning different
weights to the sub-models in the final analysis or improving the
Multi-Roberta model by adding more BERT-related embeddings, if the
device allows it.

To further evaluate the accuracy of the final output, we compared our
models' results with the \"textstat\" Python package. The package
includes readability and Dale-Chall readability level functions that
provide a numeric gauge of a text\'s comprehension difficulty.

The readability score indicates how easy or difficult a text is to
understand. The Dale-Chall readability function is a readability test
that provides a score that corresponds to the average grade level at
which the text can be understood. In the \"textstat\" package, this
function generates a score ranging from 4th grade to 15th grade, with
higher scores indicating that the text can be understood by readers with
a higher level of education. These scores can be used to evaluate the
overall quality of the writing.

![](https://github.com/hs1063/580_final/blob/main/imgs/img/image016.png){width="6.5in" height="1.5694444444444444in"}

Our model and the \"textstat\" results both indicate that the second
text has the lowest grade level and may be the most difficult to
understand. However, some texts with higher grade levels, such as those
corresponding to college level 2, may also be difficult to read due to
their use of academic writing style rather than general writing style.

In the future, our next step is to combine machine learning models, such
as BERT, with the Dale-Chall readability function to develop a more
interactive web application that can help ESL students improve their
writing skills. This application will be customized to be more efficient
and will use the \"textstat\" package combining those six writing style
features to provide students with detailed feedback on their writing. We
believe this study will become a useful tool for ESL students to improve
their writing abilities and succeed in their academic studies.

![](https://github.com/hs1063/580_final/blob/main/imgs/img/image017.png){width="6.5in" height="2.66875in"}

