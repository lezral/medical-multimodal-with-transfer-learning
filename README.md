# medical-multimodal-with-transfer-learning

Data is one of the essential ingredients to power deep learning research. Small datasets, especially specific to medical institutes, bring challenges to deep learning training stage. This work aims to develop a practical deep multimodal that can classify patients into abnormal and normal categories accurately as well as assist radiologists to detect visual and textual anomalies by locating areas of interest. The detection of the anomalies is achieved through a novel technique which extends the integrated gradients methodology with an unsupervised clustering algorithm. This technique also introduces a tuning parameter which trades off true positive signals to denoise false positive signals in the detection process. To overcome the challenges of

the small training dataset which only has 3K frontal X-ray images and medical reports in pairs, we have adopted transfer learning for the multimodal which concatenates the layers of image and text submodels. The image submodel was trained on the vast ChestX-ray14 dataset, while the text submodel transferred a pertained word embedding layer from a hospital-specific corpus. Experimental results show that our multimodal improves the accuracy of the classification by 4% and 7% on average of 50 epochs, compared to the individual text and image model, respectively.

This Notebook Collection is an overview of the work that has been done regarding the above.

This Repo works with PUBLIC dataset of the Indiana University Chest X-ray (https://openi.nlm.nih.gov/detailedresult?img=CXR111_IM-0076-1001&req=4)

Using a subsample of the data,(pre-processed) .pkl files and also all relevant IDs, raw-text and labels, vocab.json(needed for text modeling) can be found via the link:

This Repo is just a demo of the original short paper, which can also be found in the Repo.

Below is a summary of the findings;

### Experiments

#### Image

Transfer learning from the relevant features with the ChestX-ray14 and Imagenet, and fine-tuning the models with our data gave out interesting results on our test datasets. Just under a few epochs on our dataset DenseNet-121 trained on ChestX-ray14 showed promising behavior. Below are some example behaviors.

![alt text](https://github.com/faikezra/medical-multimodaling-with-transfer-learning/blob/master/repo_images/cnns.png)

#### Text

1-D Text model CNN's performance on the text data can be seen below (different models are just models built on 5 different cross-validation sets);

![alt text](https://github.com/faikezra/medical-multimodaling-with-transfer-learning/blob/master/repo_images/texts.png)

#### Multimodal

Below one can see the dominance of the Multimodal when compared with the other two submodels;

![alt text](https://github.com/faikezra/medical-multimodaling-with-transfer-learning/blob/master/repo_images/graph1.png)


### Detection

#### Image
Below one can find a sample X-ray image for the public dataset
![alt text](https://github.com/faikezra/medical-multimodaling-with-transfer-learning/blob/master/repo_images/sample_xray.png)

Making use of learned weights, and treating the model as a function, and taking the partial derivative w/ respect to the inputs (the pixels), we are able to pull some explanations from the image model. A wrapper provided in this repo localizes these gradients and draws a circle around the relevant parts of the image.

![alt text](https://github.com/faikezra/medical-multimodaling-with-transfer-learning/blob/master/repo_images/detection.png)

#### Focused Image

Due to noise a threshold implemented into the localizer trades off some signal to washout noise. Below image is a noisy x-ray explanation (on the right) vs a focused (on the left) version.
![alt text](https://github.com/faikezra/medical-multimodaling-with-transfer-learning/blob/master/repo_images/focus_detection.png)

#### Text

Let's take the ground truth of the above image that we have explained;

![alt text](https://github.com/faikezra/medical-multimodaling-with-transfer-learning/blob/master/repo_images/raw_text.png)

We treat the text block as a picture as well. Each word (after the embedding layer) is a vector and a sentence is a matrix. Once treating the of the embedding layer (as one cannot take the gradient of the text sequence and the embedding layer) as the variable and taking the gradient of the whole model, we are left with a similar 'picture detection', where every row is the explanation of the word.

Below is a representation of the detection above;

![alt text](https://github.com/faikezra/medical-multimodaling-with-transfer-learning/blob/master/repo_images/text_explainations1.png)

As plotly is interactive one can zoom in and out of areas of interest;

![alt text](https://github.com/faikezra/medical-multimodaling-with-transfer-learning/blob/master/repo_images/text_explainations.png)

As it can be seen the image focuses on the enlarged heart and the spine as the text detection compliments it as well.
