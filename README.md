# BCIProject_SleepEEGNet_Refined
- Video Link: [BCI_SleepEEGNet_Refined](https://youtu.be/XgWqZGABv24)

## Introduction
  The electroencephalogram (EEG) serves as a pivotal tool in diagnosing sleep
disorders, with sleep studies involving the continuous recording of EEG data to monitor
brain activity throughout the night. However, the manual analysis of these lengthy
recordings poses a significant challenge for sleep experts, underscoring the need for
automated sleep stage classification systems.

  Existing automated sleep scoring methods fall into two main categories: those
reliant on hand-engineered features and those leveraging deep learning for automated
feature extraction. While the former achieves reasonable results, it necessitates prior
expertise in sleep analysis and may struggle with diverse datasets. In contrast, deep
learning algorithms can autonomously extract pertinent features from raw EEG signals.
Despite the promise of deep learning in sleep stage classification, the class
imbalance issue within sleep datasets persists as a significant obstacle. This imbalance
undermines the efficacy of both traditional machine learning techniques and deep
learning approaches in achieving expert-level performance.

  To address these challenges, we proposed a machine learning model that is solely based on traditional convolution neural networks (CNNs) structure. Our model adopts a sequence-to-sequence deep learning architecture and is able to achieve an overall accuracy of 84%.


### Data Description

- **Experimental Design/Paradigm:**

    Sleep Cassette data were obtained in a 1987-1991 study of age effects on sleep in healthy Caucasians aged 25-101, without any sleep-related medication. Subjects wore a modified Walkman-like cassette-tape recorder for 20 hours each during two subsequent day-night periods for 20 hours each.
    We would use ICA, frequency filter and ASR to remove the artifact, and train the data with CNN model.

- **Procedure for Collecting Data:**

    Most young volunteers underwent the electrode placement procedure at the hospital.     
    The senior age group underwent this procedure at their residences, performed by physician and an EEG technician.

- **Hardware and Software Used:**

    Modified Oxford four-channel cassette recorder with frequency response range from 0.5 to 100Hz.       
    They did not specify which software they used. Only know that the signal was recorded on PC and Rechtschaffen and Kales sleep stages were manually scored.

- **Data Size:**

    - Raw data for one PSG.edf file is about 50MB
    - Each Hypnogram.edf file is about 5KB
    - The total raw data is 1.96GB

- **Number of Channels, Sampling Rate:**

    The one that is used in our system is Fpz-Cz data.    
    The EOG and EEG signals were sampled at 100Hz. The EMG signal was high-pass filtered, rectified and low-pass filtered. Oronasal airflow, rectal body temperature and the event marker were sampled at 1Hz.
 
- **The Source of Data and Experiment:**

    - Data source: <https://physionet.org/content/sleep-edfx/1.0.0/#files-panel> , we use the data in folder sleep-cassette.
    - Experiment: Mourtazaev, M. S., Kemp, B., Zwinderman, A. H., & Kamphuisen, H. A. (1995). Age and gender affect different characteristics of slow waves in the sleep EEG. Sleep, 18(7), 557–564. https://doi.org/10.1093/sleep/18.7.557

- **More Description:**

    - Files are named in the form sc4ssNEO-PSG.edf where ss is the subject number, and N is night. One thing to noticed is that the first nights of subjects 36 and 52, and the second night of subject 13, were lost.
    The sleep-edf database contains 197 whole-night PolySomnoGraphic sleep recordings, containing EEG, EOG, chin EMG, and event markers.
    There are two types of data. The *PSG.edf files are whole-night polysmnographic sleep recordings contains EEG, EOG, submental chin EMG, and an event marker. These files are formatted in EDF. The *HYPNOGRAM.EDF files contain annotations of the sleep patterns that correspond to the PSGs. The patterns consist of sleep stages: W, R 1, 2, 3, 4, M (Movement time) and ? (not scored). These stages were manually scored by well-trained technicians, but base on channel Fpz-Cz/Pz-Oz EEGs. These files are formatted in EDF+.

### Quality Evaluation

- **Existing Literature:**

    - Mousavi S, Afghah F, Acharya UR (2019) SleepEEGNet: Automated sleep stage scoring with sequence to sequence deep learning approach. PLOS ONE 14(5): e0216456. https://doi.org/10.1371/journal.pone.0216456:    
      The system we based on and analyze in midterm report. The output can guarantee the credibility of the data
    - B. Kemp, A. H. Zwinderman, B. Tuk, H. A. C. Kamphuisen and J. J. L. Oberye, "Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG," in IEEE Transactions on Biomedical Engineering, vol. 47, no. 9, pp. 1185-1194, Sept. 2000, doi: 10.1109/10.867928.:     
      They derived the maximum-likelihood estimator of the feedback gain and applied it to quantify sleep depth. As the results, females were found to have twice the SWP of males, but no gender effect on SW% was found. It confirms earlier reports that gender affects SWP but not sleep depth. Meaning the data is reliable.

- **ICLabel:**

  Unfortunately, even though we are able to run ICA and ASR, the ICLabel can't be established due to the lack of information of locations. The resource did not specify the hardware they utilized, thererfore the location data relocate is not possible. The paper points out that they have filtered the signal and removed the artifact (eyes, muscle, sweat).    
      Comparison of SC4001E0-PSG.edf: ASR-corrected and data after ICA.
    <img width="1440" alt="Screen Shot 2024-06-04 at 12 55 48 AM" src="https://github.com/ggbdmn/BCIProject_SleepEEGNet_Refined/assets/82556349/50884f09-e7a7-4063-bf7a-e51550dd6385">
      Comparison of SC4002E0-PSG.edf: data after ICA and ASR-corrected.
      <img width="1428" alt="Screen Shot 2024-06-09 at 12 48 06 AM" src="https://github.com/ggbdmn/BCIProject_SleepEEGNet_Refined/assets/82556349/83459aa6-ec91-44b9-a9a3-46ea49275532">

## Model Framework
- **Data Process Pipeline**
![pipeline](https://github.com/ggbdmn/BCIProject_SleepEEGNet_Refined/assets/30823131/63e293e6-c454-45d5-839b-17640bab12e6)

*The ICA part is optional.

- **The Model Architecture**
![model](https://github.com/ggbdmn/BCIProject_SleepEEGNet_Refined/assets/30823131/6404b7d6-b501-417a-a815-e543c6e532b6)

## Validation
1. **Data Preprocessing and Augmentation**
- **Normalization and Scaling:** Data normalization using MinMaxScaler ensures the EEG data is scaled appropriately, which is crucial for the convergence of neural networks.
- **Oversampling and Undersampling:** Techniques like SMOTE (Synthetic Minority Over-sampling Technique), RandomUnderSampler, and ADASYN are used to handle class imbalance, ensuring the model does not become biased towards more frequent classes.
2. **Model Training and Hyperparameter Tuning**
- **Cross-Validation:** The script uses 5-fold cross-validation (num_folds: int = 5) to ensure the model’s performance is robust and not dependent on a single train-test split.
- **Hyperparameter Tuning:** The use of hyperparameters such as epochs, batch_size, num_units, etc., are fine-tuned to optimize model performance.
3. **Model Architecture**
- **Deep Learning Layers:** Employing a combination of CNN to capture temporal dependencies and spatial features from the EEG data.
- **Dropout Layers:** Dropout layers are used for regularization to prevent overfitting.
4. **Evaluation Metrics**
- **Confusion Matrix:** Used to visualize the performance of the classification model and to calculate other metrics.
- **F1 Score:** The harmonic mean of precision and recall and is particularly useful for imbalanced classes.
- **Cohen’s Kappa Score:** Measuring the agreement between predicted and true labels, accounting for the possibility of agreement occurring by chance.
5. **Model Testing and Validation**
- **Train-Test Split:** Data is split into training and testing sets to evaluate the model's performance on unseen data.
- **Test Step Validation:** Regular testing after a defined number of epochs (test_step: int = 5) ensures that the model's performance is monitored throughout the training process.

## Usage
### Required environment:
```
python                       3.9
Package                      Version
---------------------------- -----------
absl-py                      2.1.0
astunparse                   1.6.3
certifi                      2024.2.2
charset-normalizer           3.3.2
contourpy                    1.2.1
cycler                       0.12.1
decorator                    5.1.1
grpcio                       1.64.1
h5py                         3.11.0
imbalanced-learn             0.12.3
imblearn                     0.0
importlib-metadata           7.1.0
importlib-resources          6.4.0
joblib                       1.4.2
keras                        2.15.0
libclang                     18.1.1
Markdown                     3.6
markdown-it-py               3.0.0
MarkupSafe                   2.1.5
matplotlib                   3.9.0
mdurl                        0.1.2
ml-dtypes                    0.2.0
mne                          1.7.0
namex                        0.0.8
numpy                        1.26.4
oauth2client                 4.1.3
oauthlib                     3.2.2
opt-einsum                   3.3.0
optree                       0.11.0
packaging                    24.0
pandas                       2.2.2
pillow                       10.3.0
platformdirs                 4.2.2
pooch                        1.8.1
protobuf                     4.25.3
psutil                       5.9.8
pyasn1                       0.6.0
pyasn1-modules               0.4.0
pygments                     2.18.0
pyparsing                    3.1.2
pyprep                       0.4.3
python-dateutil              2.9.0.post0
pytz                         2024.1
requests                     2.32.1
requests-oauthlib            2.0.0
rich                         13.7.1
rsa                          4.9
scikit-learn                 1.5.0
scipy                        1.13.1
setuptools                   49.2.1
six                          1.16.0
tensorboard                  2.15.2
tensorboard-data-server      0.7.2
tensorflow                   2.15.0
tensorflow-addons            0.23.0
tensorflow-estimator         2.15.0
tensorflow-io-gcs-filesystem 0.37.0
termcolor                    2.4.0
threadpoolctl                3.5.0
tqdm                         4.66.4
typeguard                    2.13.3
typing-extensions            4.12.1
tzdata                       2024.1
urllib3                      2.2.1
werkzeug                     3.0.3
wheel                        0.43.0
wrapt                        1.14.1
zipp                         3.18.2
```

### Execution Pipeline:
- **Dataset Preparation**
  + We evaluated our model using the [Physionet Sleep-EDF datasets](https://physionet.org/content/sleep-edfx/1.0.0/) published in 2013
  + Under Terminal,
  To download SC subjects from the Sleep_EDF (2013) dataset, use the below script:
  ```
  cd data_2013
  chmod +x download_physionet.sh
  ./download_physionet.sh
  ```
  + Use below scripts to extract sleep stages from the specific EEG channels of the Sleep_EDF (2013) dataset:
  ```
  python prepare_physionet.py --data_dir data_2013 --output_dir data_2013/eeg_fpz_cz --select_ch 'EEG Fpz-Cz'
  ```
  + If there is a need to go through the process of ICA, open the **seq2seq_sleep_sleep-EDF.py** file, and deannotate the line 270-306 like below:
  ```
  ...
  ################################# ICA ##########################################  
        n_samples, n_channels, n_timepoints = X_train.shape

        # Initialize arrays to store the transformed data
        X_train_ica = np.zeros_like(X_train)
        X_test_ica = np.zeros_like(X_test)

        # Define the number of components for ICA
        n_components = min(n_timepoints, n_samples)  # Choose a feasible number of components

        # Apply ICA separately on each channel
        for i in range(n_channels):
            # Extract the data for the current channel across all samples
            X_train_channel = X_train[:, i, :]  # Shape: (n_samples, n_timepoints)
            X_test_channel = X_test[:, i, :]    # Shape: (number of test samples, n_timepoints)
            
            # Apply ICA on the current channel data
            ica = FastICA(n_components=n_components, random_state=0)
            X_train_ica_channel = ica.fit_transform(X_train_channel)  # Shape: (n_samples, n_components)
            X_test_ica_channel = ica.transform(X_test_channel)        # Shape: (number of test samples, n_components)
            
            # Pad the transformed data to match the original number of timepoints
            if X_train_ica_channel.shape[1] < n_timepoints:
                # Pad with zeros
                X_train_ica_channel = np.pad(X_train_ica_channel, ((0, 0), (0, n_timepoints - n_components)), mode='constant')
                X_test_ica_channel = np.pad(X_test_ica_channel, ((0, 0), (0, n_timepoints - n_components)), mode='constant')
            
            # Assign the transformed data back to the respective place
            X_train_ica[:, i, :] = X_train_ica_channel
            X_test_ica[:, i, :] = X_test_ica_channel

        # Verify shapes
        print(X_train_ica.shape)  # Should be (n_samples, n_channels, n_timepoints)
        print(X_test_ica.shape)
        X_train = X_train_ica
        X_test = X_test_ica
        ################################# ICA ########################################## 
  ...
  ```
  then run the scripts to extract sleep stages.

- **Train**
  + Run the below script to train SleepEEGNET model using Fpz-Cz channel of the Sleep_EDF (2013) dataset:
  ```
  python seq2seq_sleep_sleep-EDF.py --data_dir data_2013/eeg_fpz_cz --output_dir output_2013
  ```
  + You may be able to adjust the parameters in the **seq2seq_sleep_sleep-EDF.py** file, line 35-46:
  ```
  class HParams:
    epochs: int = 120  # 120, 300 ###################################change
    batch_size: int = 40  # 20, 10
    num_units: int = 128
    embed_size: int = 10
    input_depth: int = 3000
    #n_channels: int = 40  ###100e
    max_time_step: int = 40  # 5 3 second best 10# 40 # 100
    output_max_length: int = 11  # max_time_step +1
    akara2017: bool = True
    test_step: int = 5  # each 10 epochs
    num_folds: int= 5  ######################################### change
    num_classes: int = 5
  ```

## Results
- **Run model without ICA (120 epochs, 20 folds)**
  + The accuracy and loss by epochs:
    ![441948826_1903231473455384_3976518203206292983_n](https://github.com/ggbdmn/BCIProject_SleepEEGNet_Refined/assets/30823131/db6327cd-09a0-46e2-a83b-2e83161149e1)
    
  + The final results of loss, accuracy, Confusion Matrix(W1, N1, N2, N3, REM), Cohen's Kappa and F1 scores:
    <img width="783" alt="截圖 2024-06-06 下午4 45 45" src="https://github.com/ggbdmn/BCIProject_SleepEEGNet_Refined/assets/30823131/38461b8d-bdcc-422a-8a6a-82b0d8da3fc7">

- **Run model with ICA (60 epochs, 5 folds)**
  + The accuracy and loss by epochs:
    ![436443770_967867174980589_3275816979153451228_n](https://github.com/ggbdmn/BCIProject_SleepEEGNet_Refined/assets/30823131/31ef9258-7879-4e50-8d80-495b9b5b19ae)
    
  + The final results of loss, accuracy, Confusion Matrix(W1, N1, N2, N3, REM), Cohen's Kappa and F1 scores:
    <img width="781" alt="截圖 2024-06-06 晚上10 47 47" src="https://github.com/ggbdmn/BCIProject_SleepEEGNet_Refined/assets/30823131/d483a76f-a8af-4cf8-b862-0c23ef86c53a">

- **Conclusion**
  + The model without ICA achieves a higher test accuracy compared to the model with ICA, indicating that using ICA in this application is not helpful, on the contrary, ICA undermines the overall performance in terms of training speed and accuracy
  + Without applying ICA receives the overall test accuracy of the model, reaching 83.9% compared to 43.4% with ICA. This suggests that ICA may remove some useful information along with the noise, negatively impacting the model's performance. Therefore, excluding ICA from the preprocessing steps results in better performance for sleep stage classification.

## References
- [github:MousaviSajad](https://github.com/MousaviSajad/SleepEEGNet)
- [SleepEEGNet: Automated Sleep Stage Scoring with
Sequence to Sequence Deep Learning Approach](https://arxiv.org/pdf/1903.02108)
