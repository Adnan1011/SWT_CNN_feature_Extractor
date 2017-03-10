function features = GenerateFeatures(imgPath, option)

img = imread(imgPath);

% Transoform to YCbCr color space
if size(img,3) ~= 1 
        temp = rgb2ycbcr(img); 
        imgL = temp(:,:,1); 
        ColorLayers = double(temp(:,:,2:3));
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
scales = 7;

switch lower(option)   
    case lower('SWT')
        features = SWT_coef(imgL,scales);
        features = features(:,:,end:-1:1);
        % Keep only the detail (horizontal, vertical and diagonal) tiles in
        % each scale
        ind = 8:4:size(features,3); 
        features(:,:,ind) = []; 
    otherwise
        error('Unknown option'); 
end

if size(img,3) ~= 1 
    features = cat(3,features,ColorLayers); 
end 
    
end 
