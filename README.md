# medical-multimodaling-with-transfer-learning

Data is one of the essential ingredients to power deep learning research. Small datasets, especially specific to medical institutes, bring challenges to deep learning training stage. This work aims to develop a practical deep multimodal that can classify patients into abnormal and normal categories accurately as well as assist radiologists to detect visual and textual anomalies by locating areas of interest. The detection of the anomalies is achieved through a novel technique which extends the integrated gradients methodology with an unsupervised clustering algorithm. This technique also introduces a tuning parameter which trades off true positive signals to denoise false positive signals in the detection process. To overcome the challenges of

the small training dataset which only has 3K frontal X-ray images and medical reports in pairs, we have adopted transfer learning for the multimodal which concatenates the layers of image and text submodels. The image submodel was trained on the vast ChestX-ray14 dataset, while the text submodel transferred a pertained word embedding layer from a hospital-specific corpus. Experimental results show that our multimodal improves the accuracy of the classification by 4% and 7% on average of 50 epochs, compared to the in dividual text and image model, respectively.

This Notebook Collection is an overview of the work that has been done regarding the above.

This Repo works with PUBLIC dataset of the Indiana University Chest X-ray (https://openi.nlm.nih.gov/detailedresult?img=CXR111_IM-0076-1001&req=4)

Using a subsample of the data,(pre-processed) .pkl files and also all relevent IDs, raw-text and labels, vocab.json(needed for text modaling) can be found via the link:

This Repo is just a demo of the original shor paper, which can also be found in the Repo.

Below is a summary of the findings;

### Experiments

#### Image

#### Text

#### Multimodal

### Detection

#### Image

#### Focused Image

#### Text
