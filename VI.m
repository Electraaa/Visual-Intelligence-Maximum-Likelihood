clear;

%% Read in image data
r = imread('r.bmp');
g = imread('g.bmp');
b = imread('b.bmp');
le = imread('le.bmp');
fe = imread('fe.bmp');
nir = imread('nir.bmp');
load('ground_truth.mat');
image_array = {r,g,b,le,fe,nir};

%% Select training samples
% Sample selection
% Programmatically select 20 samples from images based on ground truth % data.
numberOfSamples = 20;
sampleHolder = cell(1,4);
for a = 1:4
    % find index of all elements already a classified as class a
    findSamples = find(labelled_ground_truth == a);
    % randomly sample 20 of these indices
    rng(123456); %ensure reproducability
    randSample = datasample(findSamples,numberOfSamples,'Replace',false);
    samples = zeros(numberOfSamples,6);
    for j = 1:numel(image_array)
        for i = 1:numel(randSample)
            % sample at i,j = current image at the randomly select sample
            samples(i,j) = image_array{j}(randSample(i));
        end
    end
    % Store sample vector
    sampleHolder{a} = samples;
end


%% Plot gaussian distribution
pdsf = cell(1,4);
for a = 1:4
    x = sampleHolder{a};
    [m,s] = normfit(x);
    y = normpdf(x,m,s);
    pdsf{a} = y; plot(x,y,'.');
end
for a = 1:4
    figure
    plot(sampleHolder{a},pdsf{a},'.');
    title(strcat('Class', num2str(a)))
end

% Calculate covariances
covariances = cell(1,4);
for a = 1:4
    covariances{a} = cov(sampleHolder{a});
end
% Calculate means
means = cell(1,4);
for a = 1:4
    means{a} = mean(sampleHolder{a}, 1);
end


%% Calculate Maximum Likelihood
for i = 1:356 % iterate across image
    for j = 1:211
        maximumscores = cell(1,4);
        for class = 1:4 % For each class
            maxProbability = -1;
            maxProbabilityClass = 0;
            pixelarray = [];
            for layer = 1:6 % For each layer
                pixelarrays = image_array{1,layer};
                pixelarray = [pixelarray,pixelarrays(j,i)];
            end
            Xmean = means{class};
            pixelarray = double(pixelarray);
            covariance = covariances{class};
            % Apply function
            maximumscore = exp(-0.5*(pixelarray - Xmean) * inv(covariance(1:6,1:6)) * transpose(pixelarray - Xmean));
            % Store variables
            maximumscores{class} = maximumscore;
        end
        maximumscores = cell2mat(maximumscores);
        maximum = max(max(maximumscores));
        [x]=find(maximumscores==maximum); % Find index(class) with highest scoree
        if(x == labelled_ground_truth(j,i))
            maxProbabilityClass = x; % Set class to apply if it matches ground truth
        end
        Classification_Matrix(j,i) = maxProbabilityClass; % Apply class value to matrix
    end
end

%% Display results of classifcation against ground truth
colour_map = [1 0 0;0 1 0;0 0 1;0 1 1];
figure
subplot(1,2,1);
imshow(label2rgb(Classification_Matrix,colour_map))
subplot(1,2,2);
imshow(label2rgb(labelled_ground_truth,colour_map))

% Calculate Confusion matrix
target = reshape(labelled_ground_truth,1,[]);
actual = reshape(Classification_Matrix,1,[]);
[C,order] = confusionmat(target, actual);
accuracy = trace(C)/sum(sum(C))
C