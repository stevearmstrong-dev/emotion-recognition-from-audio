# Recognize Emotions from Speech
In this project, I extract features from audio files using the Librosa module and then use a classifier to recognize the emotions from the audio files.

The features I am are extracting are:

* Mel Frequency Cepstral Coefficients (MFCC).
* Chroma of the audio file.
* Spectral Scale of the pitch of the audio.

After extracting the above features from the audio file I,
1. Create different datasets for training and testing.
2. Initialize a new classifier using scikit-learn to classify the audio file features to detect emotions.
3. Finally, I compute the accuracy of the classifier.
