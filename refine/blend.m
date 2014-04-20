% Copyright (c) 2014, Truyen Tran (tranthetruyen@gmail.com)
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without modification,
% are permitted provided that the following conditions are met:
%
% - Redistributions of source code must retain the above copyright notice,
%   this list of conditions and the following disclaimer.
%
% - Redistributions in binary form must reproduce the above copyright notice,
%   this list of conditions and the following disclaimer in the documentation
%   and/or other materials provided with the distribution.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
% DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
% ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
% NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
% EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



%  	
% 	blend several models trained in the refinement phase
%

% load('label'); %the matrix of train/valid outcomes, one row per image


rng(6789);
start = tic();

testPredicts_file	= 'testPredicts.csv'; %the submission file

load('label');
[dataSize,taskSize] = size(label);

isNNet			= 1;

if 1 %load outcomes of NNets in the refinement step
	disp('Load data');

	clear trainData0;
	clear testData0;
	clear validData0;

	j = 0;
	for modelId=[1:8]
		j = j + 1;
		trainData0{j}	= load(['predicts/trainPredicts' num2str(modelId) '.csv']);
		validData0{j}	= load(['predicts/validPredicts' num2str(modelId) '.csv']);
		testData0{j}	= load(['predicts/testPredicts' num2str(modelId) '.csv']);
	end

	disp(['Loading time ' num2str(round(toc(start)))]);
end

%return;

ModelNo = length(trainData0);

if 1 %normalizing the data
	disp('Prepare data');
	
	maxId 		= 61578;
	trainIds	= [1:60416];
	
	validIds	= setdiff([1:maxId],trainIds);
	testIds		= setdiff([1:dataSize],[trainIds,validIds]);

	trainOutcomes	= label(trainIds,:);
	validOutcomes	= label(validIds,:);

	trainSize	= length(trainIds);
	validSize	= length(validIds);
	testSize	= length(testIds);

	trainData = [];
	validData = [];
	testData = [];
	
	for k=1:taskSize
		trainData{k}	= [];
		validData{k}	= [];
		testData{k}		= [];
	end
	
	for modelId=1:ModelNo
		for k=1:taskSize
			trainData{k} 	= [trainData{k}, trainData0{modelId}(:,k)];
			validData{k} 	= [validData{k}, validData0{modelId}(:,k)];
			testData{k}		= [testData{k}, testData0{modelId}(:,k)];
		end		
	end

	% data normalization
	for k=1:taskSize

		meanData1 	= mean(trainData{k});
		stdData1 	= std(trainData{k});
		
		meanData2 	= mean(testData{k});
		stdData2 	= std(testData{k});

		trainData{k}	= (trainData{k} - repmat(meanData1,trainSize,1)) ./ single(1e-5 + repmat(stdData1,trainSize,1));
		validData{k}	= (validData{k} - repmat(meanData2,validSize,1)) ./ single(1e-5 + repmat(stdData2,validSize,1));
		testData{k}		= (testData{k} - repmat(meanData2,testSize,1)) ./ single(1e-5 + repmat(stdData2,testSize,1));
	end
	
	disp(['Preparation time ' num2str(round(toc(start)))]);
end

validIds1	= [1:500];

validIds2 	= setdiff([1:validSize],validIds1);
validSize1	= length(validIds1);
validSize2 	= length(validIds2);

validOutcomes1	= validOutcomes(validIds1,:);
validOutcomes2	= validOutcomes(validIds2,:);


%regression with NNet
	
clear opt;

opt.l2_penalty		= 1e-5;
opt.report_interval	= 10;
opt.nIters			= 500;
opt.HiddenSize		= 50;    
opt.nConts			= 0; 	%continuation steps
opt.MaxNorm			= 1;	%max norm of weights coming to hidden units
opt.norm_penalty	= 1;
opt.LabelSize		= 1;

fprintf('Running NNet, single task mode...\n');

trainPredicts	= zeros(trainSize,taskSize);
validPredicts1	= zeros(validSize1,taskSize);
validPredicts2	= zeros(validSize2,taskSize);
testPredicts	= zeros(testSize,taskSize);


for k=1:taskSize

	fprintf('Outcome %modelId...\n',k);

	%easy tasks, don't have to learn, just averaging...
	if 1
		if ismember(k,[1,2,4,5,6,8,14:19,24,26,28,33])
			for modelId=1:ModelNo
				trainPredicts(:,k)	= trainPredicts(:,k) + trainData0{modelId}(:,k)/ModelNo;
				validPredicts1(:,k)	= validPredicts1(:,k) + validData0{modelId}(validIds1,k)/ModelNo;
				validPredicts2(:,k)	= validPredicts2(:,k) + validData0{modelId}(validIds2,k)/ModelNo;
				testPredicts(:,k)	= testPredicts(:,k) + testData0{modelId}(:,k)/ModelNo;
			end

			continue;
		end
	end

	BootstrapNo = 1;

	for b=1:BootstrapNo

		D = 2000;		
		if BootstrapNo == 1
			trainIds1 = [1:D];
			validIds1x = [1:validSize1];
		else
			fprintf('Bootstrap %modelId...\n',b);
			trainIds1	= randi(trainSize,1,D);
			validIds1x	= randi(validSize1,1,validSize1);
		end

		currOutcomes	= [trainOutcomes(trainIds1,k);validOutcomes1(validIds1x,k)];
		currData		= [trainData{k}(trainIds1,:);validData{k}(validIds1(validIds1x),:)];

		outMean		=  mean(currOutcomes);
		outStd		=  std(currOutcomes);

		% blending
		opt.label 	= (currOutcomes - outMean) / outStd;
		nNetStruct	= nNet(currData,[],opt,'train');

		% prediction
		trainPreds = outMean + outStd.*nNet(trainData{k},nNetStruct,opt,'test');
		trainPreds(trainPreds < 0) = 0; trainPreds(trainPreds > 1) = 1;
		trainPredicts(:,k)	= trainPredicts(:,k) + trainPreds;

		validPreds = outMean + outStd.*nNet(validData{k}(validIds1,:),nNetStruct,opt,'test');
		validPreds(validPreds < 0) = 0; validPreds(validPreds > 1) = 1;
		validPredicts1(:,k)	= validPredicts1(:,k) + validPreds;

		validPreds = outMean + outStd.*nNet(validData{k}(validIds2,:),nNetStruct,opt,'test');
		validPreds(validPreds < 0) = 0; validPreds(validPreds > 1) = 1;
		validPredicts2(:,k)	= validPredicts2(:,k) + validPreds;

		testPreds = outMean + outStd.*nNet(testData{k},nNetStruct,opt,'test');
		testPreds(testPreds < 0); testPreds(testPreds > 1) = 1;
		testPredicts(:,k)	= testPredicts(:,k) + testPreds;
	end

	trainPredicts(:,k)	= trainPredicts(:,k)/BootstrapNo;
	validPredicts1(:,k)	= validPredicts1(:,k)/BootstrapNo;
	validPredicts2(:,k)	= validPredicts2(:,k)/BootstrapNo;
	testPredicts(:,k)	= testPredicts(:,k)/BootstrapNo;
end

trainRMSE	= sqrt(sum(sum((trainOutcomes - trainPredicts).^2))/(trainSize*taskSize));
validRMSE1	= sqrt(sum(sum((validOutcomes1 - validPredicts1).^2))/(validSize1*taskSize));
validRMSE2	= sqrt(sum(sum((validOutcomes2 - validPredicts2).^2))/(validSize2*taskSize));

fprintf('RMSE, train: %.5f, valid1 %.5f, valid2 %.5f, time: %.1f\n',trainRMSE,validRMSE1,validRMSE2,toc(start));


%-- printing out the predictions -----
dlmwrite(testPredicts_file,testPredicts);


