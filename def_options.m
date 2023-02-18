options = trainingOptions('adam', ... % adam优化算法 自适应学习率 
'MaxEpochs',500,...% 最大迭代次数 
 'MiniBatchSize',10, ...%最小批处理数量 
'GradientThreshold',1, ...%防止梯度爆炸 
'InitialLearnRate',0.005, ...% 初始学习率 
'LearnRateSchedule','piecewise', ...
 'LearnRateDropPeriod',125, ...%125次后 ，学习率下降 
'LearnRateDropFactor',0.2, ...%下降因子 0.2
'ValidationData',{XTrain,YTrain}, ...
 'ValidationFrequency',5, ...%每五步验证一次 
'Verbose',1, ...
 'Plots','training-progress');