# Algorithmic Classification of Classical Piano Music by Composer

## Quickstart — Transferring Project to a New PC

1. Install python: <https://www.python.org/downloads/>
   - Verify with ``python --version``
2. Download repo from <https://github.com/JAMBallou/composer-analysis>
3. Download MAESTRO dataset at <https://magenta.withgoogle.com/datasets/maestro#dataset>, should go to ``data/maestro/``.
4. Create virtual environment: ``python -m venv .venv``
   - Activate using ``.venv/Scripts/activate``
5. Install dependencies: ``pip install -r requirements.txt``
   - If [using GPU](https://www.tensorflow.org/install/pip), also run ``pip install tensorflow[and-cuda]``
   - Verify: ``python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"``
6. Filter composers (``python -m src.preprocessing.filter_composers --top-n 12 --deduplicate best_quality --delete --update_csv --backup_csv``).
7. Extract MIDI features  (``python -m src.features.extract_midi_features``).
8. Extract audio features  (``python -m src.features.extract_audio_features``).
9. Test GPU setup:
   - ``nvidia-smi`` ([NVIDIA GPU driver](https://www.nvidia.com/download/index.aspx))
   - ``nvcc --version`` ([CUDA](https://developer.nvidia.com/cuda-downloads))
   - Install ``cuDNN`` ([cuDNN](https://developer.nvidia.com/cudnn))
   - Test config: ``python -m src.utils.gpu_config``
10. Temporal training (uses BiLSTM models): ``python -m src.training.train_temporal src/configs/temporal_base.yaml``

The following has been taken from my proposed research plan applying to the [Massachusetts Science and Engineering Fair (MSEF)](https://scifair.com/).

## Rationale (Problem)

*A few sentences to summarize your background research that supports your research problem and the need for a solution.*
*(Prompts: Why is this problem worth solving? Is there a global or societal need for this prototype/product?)*

This project combines techniques from music theory, physics, mathematics, and computer science to analyze works of classical music and ultimately classify them by composer. [Published in 2018 by Hawthorne et al.](https://arxiv.org/pdf/1810.12247), [the MAESTRO Dataset](https://magenta.withgoogle.com/datasets/maestro#dataset) — **M**IDI and **A**udio **E**dited for **S**ynchronous **TR**acks and **O**rganization — provided the data for this project, and contains both MIDI transcriptions and WAV audio files of virtuosic piano music. The mathematical analysis of the audio involved the Fourier transform, a method of decomposing a complex signal into its component sine waves, thus revealing the most influential frequencies in the sound of the signal and their amplitudes. These data, along with other techniques of analyzing the symbolic and audio files, will be used to train a machine learning model that can be used to determine the composer of works it has not seen before. Most of the research in this field centers around classification by genre (e.g. classical vs. pop) or by emotion rather than the identity of the composer.

## Engineering Goal

*What is the prototype or product that you hope to develop and the expected outcome(s) from it?*
*(Prompt: What are the design criteria and constraints for your project?)*

The goal of this project is to create a machine learning model that can successfully classify works of classical music by composer. As detailed in the [Prototype Testing](#prototype-testing) section, each of the progressively more difficult classification tasks (outlined in §III of [Design & Construction](#design--construction)) must receive accuracy, precision, recall, and $F_1$ scores of above 0.7 before moving on to the next model. For the last model, which aims to classify the 14 composers with the most works represented in the MAESTRO Dataset, these four performance metrics will be optimized to achieve an $F_1$ score approaching or exceeding 0.9, consistent with state-of-the-art results in related tasks. If this turns out trivial using both the recording and MIDI transcription, a final attempt will be made at classifying the 14 composers using just the WAV audio. A secondary goal of this project is to make the models used easily scalable to add more composers and works. This involves qualitative analysis of the code and how easy it is to add more composers to the system as experimentation progresses. If successful, this system could support automated musicological analysis, recommendation systems, or digital music archiving.

## Prototype Development

*Review [page 12-14 of the MSEF manual](https://scifair.com/wp-content/uploads/MSEF-HS-Manual-2024-25.pdf) to see if there are specific guidelines, safety, or forms related to the materials you are using or the equipment you are using.*
*If you determine that anything you are planning is listed as “**requires pre-approval**”, you will be prompted to complete an electronic version of the form below. If you have completed this template, most of the required information will be ready for you to copy and paste when you create your project in zFairs!*

- *[Risk Assessment Form 3 (preview-only)](https://sspcdn.blob.core.windows.net/files/Documents/SEP/ISEF/2023/Forms/3-Risk-Assessment.pdf)*

*This portion can be subdivided into the sections below:*

### Safety

*These steps will vary according to the types of projects.  Identify any potential risks and how they are addressed/minimized in your methods.  Refer to all safety equipment used, including, but not limited to, goggles, gloves, closed toe shoes, working conditions (fume hood, fire extinguisher if combustion is possible) and supervision.*

N/A

### Materials & Equipment

*Include, in list format specific names and concentrations of chemicals, equipment used, locations, how materials are obtained, etc.*

See [bibliography](#bibliography) for dataset information.

### Design & Construction

*Sequentially numbered steps that cover the production of the prototype from beginning to end. The steps should be detailed enough for someone else to be able to replicate from your directions.*

I. **Preprocessing** - In this stage, the data will be processed to remove unwanted data, and be put into a format most useful for the purposes of this project.

1. The dataset chosen for this project is the **MAESTRO Dataset**. It consists of WAV audio and MIDI files gathered during the International Piano e-Competition, in which virtuoso pianists play on contest-quality Yamaha Disklaviers that automatically transcribe the notes being played and can wire the performance to another Disklavier or to a computer file. Included with the audio and MIDI files is data about the pedal positions and striking velocity on each key of the piano for each time frame. The most updated version of the dataset contains 1276 performances from 60 classical composers, nearly 200 hours of audio. [The full metadata for the dataset can be found here](https://github.com/JAMBallou/composer-analysis/blob/main/data/maestro/maestro-v3.0.0.csv).

2. The MAESTRO Dataset is enormous, and not all of it will be used in this project. First, only the composers with a significant number of works will be included in this study. All the data from those composers represented by fewer than 25 works in the dataset will be removed, including all the works with attributions to more than one composer. [A full list of composers in the dataset by number of works can be found here](https://github.com/JAMBallou/composer-analysis/blob/main/data/maestro/composer_count.csv).

3. The 14 composers with more than 25 works in the MAESTRO Dataset, ordered by number of works:

   | **Composer** | **Number of Works in Dataset** |
   | ---------- | ------- |
   | *Frédéric Chopin* | 201 |
   | *Franz Schubert* | 186 |
   | *Ludwig van Beethoven* | 146 |
   | *Johann Sebastian Bach* | 145 |
   | *Franz Liszt* | 131 |
   | Sergei Rachmaninoff | 59 |
   | Robert Schumann | 49 |
   | Claude Debussy | 45 |
   | Joseph Haydn | 40 |
   | Wolfgang Amadeus Mozart | 38 |
   | Alexander Scriabin | 35 |
   | Domenico Scarlatti | 31 |
   | Felix Mendelssohn | 28 |
   | Johannes Brahms | 26 |

4. To aid efficiency in the model, a master JSON file will be created with information about each of the 14 above composers, including their name, the number of works represented in the dataset, and the era they composed in (i.e. Baroque, Classical, or Romantic). Beethoven is the only composer who is tricky to classify, as his early music is clearly Classical, and his later music marked the start of the Romantic period.

5. To standardize the length of each piece to 60 s, the following procedure will be used (adapted from [Costa et al.](https://www.researchgate.net/publication/311650413_An_Evaluation_of_Convolutional_Neural_Networks_for_Music_Classification_Using_Spectrograms)):

- For pieces with a runtime of greater than $90 \textrm{s}$, the interval used will be $[30 \textrm{s}, 90 \textrm{s}]$.
- For pieces with a runtime between $60 \textrm{s}$ and $90 \textrm{s}$, the endpoints will be the midpoint of the audio file plus or minus $30 \textrm{s}$.
- Pieces with a runtime shorter than $60 \textrm{s}$ will be cut from the dataset.

II. **Feature Selection** - This is the most important part of the process, and ultimately determines the accuracy of the model. This is also the step that largely differentiates this project from previous ones. For these reasons, this stage will also see the most modification as the experimentation progresses.

1. The most influential feature will be the **spectrogram**, the result of a short time Fourier transform taken on the audio file. Initially, $23 \textrm{ms}$ windows will be used with 50% overlap and a Hann window function, but these parameters may be changed throughout the course of experimentation depending on performance.

2. The Python library **pretty_midi** will be used to extract features from the MIDI files also present in the dataset alongside the WAV recordings. pretty_midi can be used to extract such things as tempi, time signature, key, and note duration (by subtracting initial time from final time).

3. The **librosa** Python library contains a number of helpful functions for extracting features from audio signals:

4. From the computed STFT, a set of **Mel-frequency cepstral coefficients (MFCC)** — a way of extracting data from an audio signal that correlates well with the human perception of sound — will be found. An MFCC takes the shape of a 39-dimensional vector. It will be calculated using  `librosa.feature.mfcc()`.

5. The **Harmonic Pitch Class Profile (HPCP)** — also known simply as the chroma — provides a visualization of the distribution of energy across the twelve pitch classes for an audio signal, encoding the harmonic content of the signal.

III. **Model Architecture** - The following types of machine learning models will likely be used in some capacity, though the exact parameters will almost certainly be tweaked.

1. **Convolutional Neural Network (CNN)** - Commonly used in image processing and computer vision. Centers around a mathematical operation called a convolution, essentially a small square set of pixels, each with a distinct weight, that, when passed over the original image, captures some piece of information about that image, like edges or dark spots. The machine learning model optimizes these squares of pixels, called kernels. CNNs will likely be used to process the Fourier or mel spectrograms and interpret the audio signal.

2. **Bidirectional Long Short-Term Memory (BiLSTM)** - An LSTM is a type of recursive neural network in which the model has a sort of “memory” of previous cases that it can “forget” if they turn out to be irrelevant. The bidirectional part means that there are two LSTMs moving through the dataset in opposite directions that can interact with one another. The BiLSTM is what will add a temporal dimension to the model. At each timestep, the Fourier transform will be taken or MFCCs will be calculated, and those results will be fed into the BiLSTM, which can use data from previous timesteps to find trends in the signal.

3. **Dense Layer** - A simple neural network where each input directly corresponds to one output. Essentially a trainable linear transformation on the input data. Dense layers will be used to normalize initial and final data, as well as between different types of layers.

4. **Deep Neural Network (DNN)** - Neural network with more than two hidden layers. Often yields state-of-the-art results, but has a tendency to overfit to the data. A DNN will be used to optimize the final model.

IV. **Methodology** - Four separate trials of increasing difficulty will be performed to optimize the code being used as the project progresses:

1. **Period Classification.** The model will attempt to classify works by period, either baroque, classical, or romantic. This should be relatively straightforward for the model because the differences between each period are so stark and there are so many works present from each period. This stage will also establish a performance baseline and begin to distinguish between broad stylistic features.

2. **Contrasting Composers.** Next, a distinction will be made between the composers. Instead of using the entire subset of 14 composers, however, two composers with a large number of works included from different periods (e.g. Bach and Chopin) will be used to further improve the model. This should also be fairly trivial.

3. **Similar Composers.** To further improve the model, the next trial will be to differentiate between two composers of the same period, which are therefore much more similar than two composers of different periods. Chopin and Schubert will be used because they have large numbers of works and are both from the Romantic period.

4. **Full Subset of MAESTRO.** Finally comes the final objective: classify each work in the subset of the MAESTRO Dataset as coming from one of these 14 composers. This is difficult not only because of the similarities between the works of many of these composers, but also because of the small amount of works some of the composers toward the end of the list have. For this reason, the goal may be modified, depending on how well the model performs.

- As seen in the [list of composers above](#design--construction), there is a precipitous drop off in the number of works between Liszt and Rachmaninoff. Therefore, this phase will be broken into two steps in practice, first using the 5 composers with more than 100 works (in italics), then phasing in the rest.

V. **Performance Evaluation**

1. A **confusion matrix** is a data visualization tool that will be employed to find the most confused composers (examples can be seen [here on ScienceDirect](https://www.sciencedirect.com/topics/engineering/confusion-matrix)). The rows of the chart are the actual classification of each composer, and the columns the predicted classification. The correctly classified cases run diagonally down the center of the chart, and high values outside that diagonal indicate that the composer in the given row was often misclassified as the composer in the given column, calling for a change in features or model architecture.

2. To ensure the model does not overfit to the data, **k-fold cross validation** will be implemented in addition to the testing subset of the data.

3. The [design criteria below](#prototype-testing) will also be used to evaluate the performance of the model.

### Prototype Testing

*Describe the steps and measurements involved in the testing of the prototype.  Include a description of the design criteria that will be employed to analyze and discuss the results of the prototype testing.*

**Design Criteria:**

**Performance** - Each of the following metrics help to determine how well the model classified the data. A score of at least 0.7 will be achieved for each of the phases of the above methodology before moving on.

- **Accuracy** is simply the percentage of items that were classified correctly by the model.
- **Precision** measures the fraction of positive predictions that are correct. High precision models avoid false positives (i.e. identifying one composer’s work as belonging to another).
- **Recall** measures the ability of a model to find relevant cases (i.e. positives). High recall models avoid false negatives (i.e. misclassifying the works of one specific composer).
- The **$F_1$ score** (also F-score or F-measure) is a measure of predictive performance for a machine learning model between 0 and 1. It is the harmonic mean of precision — the number of retrieved items that are relevant — and the recall — the number of relevant items retrieved. $F_1=\frac{2TP}{2TP+FP+FN}$,where $TP$ is true positive, $FP$ false positives, and $FN$ false negatives. The $F_1$ score is a more accurate metric of performance than accuracy when one class is more common than another.

**Scalability**
One key quality of the model that will be assessed is how easily more composers can be added to the dataset. This will be assessed qualitatively when the number of composers is increased from 5 to 14.

### Clean-up and Disposal

N/A

## Bibliography

*Key sources on your topic, from your literature review and/or background research, that helped you write this plan.  APA format is recommended.*

### Dataset

- Hawthorne, Curtis, et al. “Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset.” International Conference on Learning Representations, 2018. arXiv, <https://arxiv.org/pdf/1810.12247>. Accessed 13 October 2025.
Downloaded using Google Magenta.

### Key Sources

- Chani, Hadhrami Ab, et al. “A review on sparse Fast Fourier Transform applications in image processing.” International Journal of Electrical and Computer Engineering (IJECE), 2020. Institute of Advanced Engineering and Science (IAES), <https://ijece.iaescore.com/index.php/IJECE/article/view/20038>. Accessed 26 September 2025.
- Cooley, James W., and John Tuckey. “An algorithm for the machine calculation of complex Fourier series.” Mathematics of Computation, vol. 19, no. 90, 1965, pp. 297-301. American Mathematical Society, <https://www.ams.org/journals/mcom/1965-19-090/S0025-5718-1965-0178586-1/S0025-5718-1965-0178586-1.pdf>. Accessed 8 September 2025.
- Costa, Yandre M.G., et al. “An Evaluation of Convolutional Neural Networks for Music Classification Using Spectrograms.” Applied Soft Computing, vol. 52, Mar. 2017, pp. 28–38, <https://doi.org/10.1016/j.asoc.2016.12.024>. Accessed 23 November 2025.
- Deepaisarn, Somrudee, et al. “NLP-based music processing for composer classification.” Scientific Reports, vol. 13, no. 13228, 2023. <https://doi.org/10.1038/s41598-023-40332-0>. Accessed 13 October 2025.
- Fletcher, Neville H., and Thomas D. Rossing. The Physics of Musical Instruments. 2nd ed., New York, Springer-Verlag, Inc., 1998.
- Lenssen, Nathan, and Deanna Needell. “An Introduction to Fourier Analysis with Applications to Music.” Journal of Humanistic Mathematics, vol. 4, no. 1, 2014, pp. 72-91. The Claremont Colleges, <https://scholarship.claremont.edu/cgi/viewcontent.cgi?article=1142&context=jhm>. Accessed 8 September 2025.

### Python Libraries (in addition to Python Standard Libraries)

- Abadi, Martín, et al. TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems. Version 2.18, Google Research, 2015–2025, <https://www.tensorflow.org/>.
- Chollet, François, et al. Keras: The Python Deep Learning Library. Version 3.3, Keras.io, 2015–2025, <https://keras.io/>.
- Harris, Charles R., et al. NumPy: Fundamental Algorithms for Scientific Computing in Python. Version 2.1, NumPy Developers, 2025, <https://numpy.org/>.
- Hunter, John D., et al. Matplotlib: Visualization with Python. Version 3.9, matplotlib.org, 2025, <https://matplotlib.org/>.
- McFee, Brian, et al. librosa: Audio and Music Signal Analysis in Python. Version 0.10, librosa.org, 2025, <https://librosa.org/>.
- Raffel, Colin, and Daniel P. W. Ellis. pretty_midi: Tools and Data Structures for MIDI Processing in Python. Version 0.2, Columbia University, 2014, <https://github.com/craffel/pretty-midi>.

## Summary or Addendum

*This section is only necessary if experimentation changed through the course of the research*
*If additional SRC or IRB approval was needed, you must also provide a letter from the SRC, explaining the changes, which is then signed and dated.*
