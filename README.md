Kaggle Galaxy Zoo challenge
===========================

This code helps to land the third place in [Galaxy Zoo challenge](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge). The convolutional part uses cuda-convnet library of Alex Krizhevsky. Please see [the package](https://code.google.com/p/cuda-convnet/) first. The code was run on Windows 64bit machines. Please make sure *python*, *numpy*, *scipy*, *PIL-image* and *matlab* are available.

There are several main steps:

	1. Split randomly the training data into model development (98%) and validation subsets (2%)
	2. Extract features using ConvNet on all data
	3. Train multiple neural networks using a small portion of development data and about half of validation data
	4. Blend multiple neural networks using another neural network
	5. Average several models from steps 2, 3 and 4. Models chosen are based on their performance on the public leaderboard.

The details are described as below:

Step 1
===================================================================================================

Retrieving data
---------------

Download *images_training_rev1.zip*, *images_test_rev1.zip*, *training_solutions_rev1.zip*, and *central_pixel_benchmark.zip* from the [data page](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data). Extract all archives, rename *central_pixel_benchmark.csv* to *kaggle_submission.csv* and store them in the `./raw_data/` directory in the following way:

	raw_data/
		images_training_rev1/
			100008.jpg
			...
			999967.jpg
		images_test_rev1/
			100018.jpg
			...
			999996.jpg
		training_solutions_rev1.csv
		kaggle_submission.csv

Preparing
-----------

From root directory, run the following command in `cmd`:

	python.exe make_batches.py 1 0 0 0

This produces 140 batches data (1-59: development; 60-61: validation; 62-140: testing):
	
	RUN/
		data/
			data_batch_1
			...
			data_batch_140
			batches.meta

Step 2
===================================================================================================

Training
--------

Start training the network:

	python.exe ./cuda_convnet/convnet.py --data-path=./RUN/data/ --save-path=./RUN/model/ --test-range=60-61 --train-range=1-59 --layer-def=./model_config/gz.cfg --layer-params=./model_config/gz_param.cfg --data-provider=kaggle-galaxy-zoo-128-cropped-x90rot-zoom-memory --test-freq=590 --test-one=0 --crop-border=4 --epochs=50 --max-filesize=100000

Go to the model directory `./RUN/model/` and grasp the directory name, e.g., *ConvNet__2014-03-29_07.47.35*, which contains saved checkpoints of the model. Then keep training to 140 epochs:

	python.exe ./cuda_convnet/convnet.py -f ./RUN/model/ConvNet__2014-03-29_07.47.35 --epochs=140

Decay learning rates by 0.1 and run up to 166 epochs

	python.exe ./cuda_convnet/convnet.py -f ./RUN/model/ConvNet__2014-03-29_07.47.35 --layer-params=./model_config/gz_param_1.cfg --test-freq=118 --epochs=166

Decay learning rates by 0.1 and run up to 174 epochs

	python.exe ./cuda_convnet/convnet.py -f ./RUN/model/ConvNet__2014-03-29_07.47.35 --layer-params=./model_config/gz_param_2.cfg --epochs=174

Decay learning rates by 0.1 and run up to 182 epochs

	python.exe ./cuda_convnet/convnet.py -f ./RUN/model/ConvNet__2014-03-29_07.47.35 --layer-params=./model_config/gz_param_3.cfg --epochs=182

The training phase will take about 31.5 hours on NVIDIA Tesla M2070 GPU.

Applying
--------

Write predicted values for testing data (batch 62-140):

	python.exe ./cuda_convnet/shownet.py -f ./RUN/model/ConvNet__2014-03-29_07.47.35/182.59 --data-path=./RUN/data/ --write-features=fc37 --feature-path=./RUN/model/ConvNet__2014-03-29_07.47.35_182.59/feat_62_140_fc37 --train-range=61 --test-range=62-140 --multiview-test=1

Extract features for the next step:

	python.exe ./cuda_convnet/shownet.py -f ./RUN/model/ConvNet__2014-03-29_07.47.35/182.59 --data-path=./RUN/data/ --write-features=dropout2 --feature-path=./RUN/model/ConvNet__2014-03-29_07.47.35_182.59/feat_1_140_fc2048 --test-range=1-140 --multiview-test=1
	python.exe ./write_mat.py ./RUN/model/ConvNet__2014-03-29_07.47.35_182.59/feat_1_140_fc2048 ./refine/convnet_features.mat

Write CSV result file to submit to kaggle.
	
	python.exe ./write_result.py 62 140

The submission file - 0.07939.csv - is saved into: `./RUN/model/ConvNet__2014-03-29_07.47.35_182.59/feat_62_140_fc37/0.07939.csv`.
	

Step 3
===================================================================================================

This step and the next are run in *matlab* (tested on version 2013)

3a. Load the features generated from the ConvNet in the file `./refine/convnet_features.mat`

3b. Run `refine/refine.m` several times to create multiple models, see `refine/models.txt` for details

Step 4
===================================================================================================

Run `refine/blend.m` to blend multiple models learned in step 3b
	

Step 5
===================================================================================================

Finally, average 4 results in directory `./RUN/avg_res/` to have the final result for submission `./RUN/avg_res/final_submission.csv`:

	python.exe avg_result.py

The final submission achieved **0.07869** on the private scoreboard.