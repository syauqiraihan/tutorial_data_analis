% untuk interval 45 menit (hari cerah)

Generationdata = readtable('D:\Generation data (2).xlsx');
Irradiationdata = readtable('D:\Irradiation data (2).xlsx');

TrainIp = table2array(Irradiationdata(25:3:2521,14:-2:8));        % Membaca data iradiasi
TestIp = table2array(Generationdata(4:3:2500,14:-2:8));           % Membaca data generasi

% Menghapus nilai NaN dari data
TestIp(isnan(TrainIp)) = [];                                      
TrainIp(isnan(TrainIp)) = [];                                     
TestIp(isnan(TestIp)) = [];                                       
TrainIp(isnan(TestIp)) = [];                                       

% Menghapus noise (nilai lebih dari 50 atau kurang dari 0)
TrainIp(TestIp > 50) = [];
TestIp(TestIp > 50) = [];
TrainIp(TestIp <= 0) = [];
TestIp(TestIp <= 0) = [];
TrainIp(TrainIp > 2000) = [];
TestIp(TrainIp > 2000) = [];
TestIp(TrainIp <= 0) = [];
TrainIp(TrainIp <= 0) = [];

% Menghitung nilai minimum dan maksimum dari data
mn = min(TrainIp);
mx = max(TrainIp);
mn2 = min(TestIp);
mx2 = max(TestIp);

numTimeStepsTrain = numel(TrainIp); 

% Normalisasi data
XTrainIp = (TrainIp - mn) / (mx - mn);       
XTestIp = (TestIp - mn2) / (mx2 - mn2);

% Menampilkan grafik data yang sudah dinormalisasi
figure
plot(XTrainIp(1:16))
hold on
plot(XTestIp(1:16), '.-')
legend(["Input" "Target"])
ylabel("Irradiationdata/Generationdata")
xlabel("Waktu (interval 45 menit)")
title("Unit Generation")

% Data untuk hari cerah 1 April
YTrainIp = table2array(Irradiationdata(25:3:121,16));         % Data input uji
YTestIp = table2array(Generationdata(4:3:100,16));            % Data target uji

% Menghapus nilai NaN dari data
YTestIp(isnan(YTrainIp)) = [];
YTrainIp(isnan(YTrainIp)) = [];
YTestIp(isnan(YTestIp)) = [];
YTrainIp(isnan(YTestIp)) = [];

% Menghapus noise (nilai lebih dari 50 atau kurang dari 0)
YTrainIp(YTestIp > 50) = [];
YTestIp(YTestIp > 50) = [];
YTrainIp(YTestIp <= 0) = [];
YTestIp(YTestIp <= 0) = [];
YTrainIp(YTrainIp > 2000) = [];
YTestIp(YTrainIp > 2000) = [];
YTestIp(YTrainIp <= 0) = [];
YTrainIp(YTrainIp <= 0) = [];

YTrainIp = YTrainIp';      % Ubah baris ke kolom
YTestIp = YTestIp';        % Ubah baris ke kolom

% Menghitung nilai minimum dan maksimum dari data
mn3 = min(YTrainIp);
mx3 = max(YTrainIp);
mn4 = min(YTestIp);
mx4 = max(YTestIp);

% Normalisasi data
YTrainIp = (YTrainIp - mn3) / (mx3 - mn3);
YTestIp = (YTestIp - mn4) / (mx4 - mn4);

numFeatures = 2;           % Jumlah input
numResponses = 1;          % Jumlah output
numHiddenUnits = 200;      % Jumlah unit tersembunyi

% Definisi layer GRU (mengganti LSTM dengan GRU)
layers = [ ...
    sequenceInputLayer(numFeatures)
    gruLayer(numHiddenUnits)         % Menggunakan GRU
    fullyConnectedLayer(numResponses)
    regressionLayer];                % Lapisan regresi

% Opsi untuk pelatihan jaringan
options = trainingOptions('adam', ...
    'MaxEpochs', 250, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', 0.005, ...
    'MiniBatchSize', 50, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 90, ...
    'LearnRateDropFactor', 0.2, ...
    'Verbose', false, ...
    'Plots', 'training-progress');    % Opsi lainnya

% Melatih jaringan GRU
net = trainNetwork([XTrainIp(1:end-1); XTestIp(1:end-1)], XTestIp(2:end), layers, options); 
net = predictAndUpdateState(net, [XTrainIp(1:end-1); XTestIp(1:end-1)]);
[net, YPred] = predictAndUpdateState(net, [XTrainIp(end-1); XTestIp(end-1)]);  % Prediksi data terakhir

numTimeStepsTest = numel(YTestIp);

% Prediksi data selanjutnya dan update jaringan
for i = 2:numTimeStepsTest
    [net, YPred(:, i)] = predictAndUpdateState(net, [YTrainIp(i-1); YPred(:, i-1)], 'ExecutionEnvironment', 'cpu');
end

% Denormalisasi data prediksi
YPred = (mx4 - mn4) * YPred + mn4;      
YTest = (mx4 - mn4) * YTestIp(1:end) + mn4;

% Menghitung berbagai jenis error
PR_rmse = sqrt((YPred - YTest).^2) ./ YTest * 100;
Percentage_rmse = mean(PR_rmse);
rmse = sqrt(mean((YPred - YTest).^2));
perf = mae(YPred, YTest);
perf2 = mse(YPred, YTest);
PRerror = abs(YPred - YTest) ./ YTest;
MAPE = mean(abs(YPred - YTest) ./ YTest);

% Denormalisasi input data uji
XTestIp2 = (mx2 - mn2) * XTestIp + mn2;  

% Plot hasil prediksi dan observasi
figure
plot(XTestIp2(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain + numTimeStepsTest);
plot(idx, [XTestIp2(numTimeStepsTrain) YPred], '.-')
hold off
xlabel("Waktu (interval 45 menit)")
ylabel("KWh")
title("Peramalan untuk hari cerah")
legend(["Teramati" "Peramalan"])

% Plot hasil perbandingan prediksi dan observasi
figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred, '.-')
hold off
legend(["Teramati" "Peramalan"])
xlabel("Waktu (interval 45 menit)")
ylabel("KWh")
title("Peramalan untuk hari cerah")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Waktu (interval 45 menit)")
ylabel("KWh")
title("RMSE = " + rmse + "  percentage rmse = " + Percentage_rmse + "%  MAE = " + perf + "  MAPE = " + MAPE)

% Reset dan latih ulang jaringan
net = resetState(net);
net = predictAndUpdateState(net, [XTrainIp(1:end-1); XTestIp(1:end-1)]);
YPred = [];
numTimeStepsTest = numel(YTrainIp - 1);

% Prediksi ulang dengan input baru
for i = 1:numTimeStepsTest
    [net, YPred(:, i)] = predictAndUpdateState(net, [YTrainIp(:, i); YTestIp(:, i)], 'ExecutionEnvironment', 'cpu');
end

% Denormalisasi hasil prediksi
YPred = (mx4 - mn4) * YPred + mn4;

% Menghitung berbagai jenis error
PR_rmse = sqrt((YPred - YTest).^2) ./ YTest * 100;
Percentage_rmse = mean(PR_rmse);
rmse = sqrt(mean((YPred - YTest).^2));
perf = mae(YPred, YTest);
perf2 = mse(YPred, YTest);
PRerror = abs(YPred - YTest) ./ YTest;
MAPE = mean(abs(YPred - YTest) ./ YTest);

% Plot hasil peramalan setelah update
figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred, '.-')
hold off
legend(["Teramati" "Diprediksi"])
xlabel("Waktu (interval 45 menit)")
ylabel("KWh")
title("Peramalan dengan Update untuk hari cerah")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Waktu (interval 45 menit)")
ylabel("Error")
title("RMSE = " + rmse + "  percentage rmse = " + Percentage_rmse + "%  MAE = " + perf + "  MAPE = " + MAPE)
