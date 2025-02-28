% 带 PlutoSDR 的多通道 FM 收音机接收器
clc;
clear variables;
close all;

% SDR 和 FM 参数
centerFreq = 104.5e6;   % 中心频率（根据您当地的 FM 电台进行调整）
audioFs = 24000;     % 音频采样率
sampleRate = 2*audioFs*60; % 采样率
stations = [103.5e6, 104.3e6, 104.9e6, 105.5e6]; % 要接收的 FM 电台（调整频率）
selectedStation = 3; % 要实时播放的电台索引

% 初始化 PlutoSDR
radio = sdrrx('Pluto', 'CenterFrequency', centerFreq, ...
    'BasebandSampleRate', sampleRate,  ...
    'OutputDataType', 'single');

% 为每个电台创建 FM 解调器对象
demodFM = cell(length(stations), 1);
% 创建音频文件写入器
audioFileWriter = dsp.AudioFileWriter('merged_stations.wav', ...
    'SampleRate', audioFs);
% 初始化音频缓冲区
audioBuffer = cell(length(stations), 1);

for i = 1:length(stations)
    demodFM{i} = comm.FMBroadcastDemodulator('SampleRate', sampleRate, ...
        'AudioSampleRate', audioFs, 'Stereo', false, 'PlaySound', false);
end

% 用于实时收听的音频播放器
player = audioDeviceWriter('SampleRate', audioFs);

try
    % % 设置用于绘制频谱的图形
    % hFig = figure('Name', 'Data Spectrum', 'NumberTitle', 'off');

    % % 为原始信号创建子图
    % subplot(length(stations)+1, 1, 1);
    % hPlotOrig = plot(nan, nan);
    % title('原始信号频谱');
    % xlabel('频率 （Hz）');
    % ylabel('幅度');
    % grid on;

    % % 为过滤后的信号创建子图
    % hPlotFiltered = cell(length(stations), 1);
    % for i = 1:length(stations)
    %     subplot(length(stations)+1, 1, i+1);
    %     hPlotFiltered{i} = plot(nan, nan);
    %     title(['过滤后信号 - 电台频率', num2str(stations(i)/1e6), ' MHz']);
    %     xlabel('频率 （Hz）');
    %     ylabel('幅度');
    %     grid on;
    % end

    % 设计低通滤波器
    lpfOrder = 32;
    lpfCutoff = 0.2e6;
    b = fir1(lpfOrder, lpfCutoff/(sampleRate/2));

    % 主循环
    while true
        % 从 SDR 接收数据
        data = radio();
        L = length(data);

        % % 计算和绘制原始信号频谱
        % Y = fftshift(fft(data));
        % f = linspace(-sampleRate/2, sampleRate/2, L);
        % set(hPlotOrig, 'XData', f, 'YData', abs(Y));

        % 处理每个电台
        for i = 1:length(stations)
            % 频移
            offset = stations(i) - centerFreq;
            t = (0:L-1)'/sampleRate;
            shiftedData = data .* exp(-1i*2*pi*offset*t);

            % 应用低通滤波器
            filteredData = filter(b, 1, shiftedData);

            % % 绘制滤波后的信号频谱
            % Y_filtered = fftshift(fft(filteredData));
            % set(hPlotFiltered{i}, 'XData', f, 'YData', abs(Y_filtered));

            % 确保数据长度是抽取因子的倍数
            decFactor = 240;
            newLen = floor(length(filteredData)/decFactor)*decFactor;
            filteredData = filteredData(1:newLen);

            % FM 解调
            audio = demodFM{i}(filteredData);

            % 暂存音频数据
            audioBuffer{i} = audio;
            
            % 如果是最后一个电台，合并并写入所有声道
            if i == length(stations)
                % 确保所有缓冲区长度相同
                minLen = min(cellfun(@length, audioBuffer));
                mergedAudio = zeros(minLen, length(stations));
                for j = 1:length(stations)
                    mergedAudio(:,j) = audioBuffer{j}(1:minLen);
                end
                
                % 写入合并后的音频
                audioFileWriter(mergedAudio);
                clear audioBuffer;
            end

            % 播放所选电台
            if i == selectedStation
                player(audio);
            end
        end
        drawnow;
    end
catch ME
    % Clean up
    release(radio);
    release(audioFileWriter);
    release(player);
    rethrow(ME);
end
