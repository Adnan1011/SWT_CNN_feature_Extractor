function features = GenerateFeatures(imgPath, option, image_mat)

if ~exist('image_mat', 'var')
    img = imread(imgPath);
else
    img = image_mat;
end
% I = rgb2ycbcr(img);
% features = double(I(:,:,1)) - 110.0;
% mean_filter = (1.0 / 9.0) * ones(3, 3);
% mean_I = conv2(I, mean_filter, 'same');
% std_I = real(sqrt(conv2(I.^2, mean_filter, 'same') - (mean_I.^2) + 1e-8));
% features = single((I - mean_I) ./ std_I);
% Transform to YCbCr color space
if size(img,3) ~= 1 
        temp = double(rgb2ycbcr(img));
        imgL = temp(:, :, 1);
%         ColorLayers(:, :, 1) = temp(:, :, 2) - mean(mean(temp(:, :, 2)));
%         ColorLayers(:, :, 2) = temp(:, :, 3) - mean(mean(temp(:, :, 3)));
        clear temp
else 
    imgL = img;
end

% Transform to Lab color space
% if size(img,3) ~= 1 
%         cform = makecform('srgb2lab'); 
%         temp = applycform(img,cform);
%         imgL = temp(:,:,1);%Luminance Channel
%         ColorLayers = temp(:,:,2:3);
% end 

%scales = floor(log2(min([size(img,1),size(img,2)]))) - 1;
scales = 2;

switch lower(option)   
    case lower('SWT')
        features = SWT_coef(imgL,scales);
        features = features(:,:,end:-1:1);
        % Keep only the detail (horizontal, vertical and diagonal) tiles in
        % each scale        
        ind = 4:4:size(features,3); 
        features(:,:,ind) = []; 
        features = max(features, [], 3);
        min_feature_val = min(min(features));
        max_feature_val = max(max(features));
        features = (features - min_feature_val) / (max_feature_val - min_feature_val);        
    otherwise
        error('Unknown option'); 
end

%features = imgL;

% if size(img,3) ~= 1 
%     features = cat(3,features,ColorLayers); 
% end 
    
end 
