clc; clear; close all;

img = imread('deneme.png');
gray_img = rgb2gray(img);

% 1) Viola-Jones ile ilk tespit (tum yuzler)
faceDetector = vision.CascadeObjectDetector('FrontalFaceCART');
rawFaces = double(faceDetector(gray_img));

% Videodaki gibi her adayi goz, burun ve ten bilgisiyle dogrula.
eyeBig = vision.CascadeObjectDetector('EyePairBig');
eyeBig.UseROI = true;
eyeSmall = vision.CascadeObjectDetector('EyePairSmall');
eyeSmall.UseROI = true;
noseDetector = vision.CascadeObjectDetector('Nose');
noseDetector.UseROI = true;

valid = false(size(rawFaces,1),1);
for i = 1:size(rawFaces,1)
    valid(i) = validateFace(img, gray_img, rawFaces(i,:), ...
        eyeBig, eyeSmall, noseDetector);
end
faces = rawFaces(valid,:);

% 2) Her yuz icin iyilestirme
refinedBoxes = zeros(size(faces,1), 4);

for i = 1:size(faces,1)
    bbox = faces(i,:);
    x = bbox(1); y = bbox(2); w = bbox(3); h = bbox(4);

    rx1 = max(1, round(x + 0.10*w));
    rx2 = min(size(img,2), round(x + 0.90*w));
    ry1 = max(1, round(y + 0.05*h));
    ry2 = min(size(img,1), round(y + h));
    roi = img(ry1:ry2, rx1:rx2, :);

    ycbcr = rgb2ycbcr(roi);
    Cb = ycbcr(:,:,2);
    Cr = ycbcr(:,:,3);
    skin = (Cb >= 77 & Cb <= 127) & (Cr >= 133 & Cr <= 173);
    skin = imopen(skin, strel('disk', 2));
    skin = imclose(skin, strel('disk', 5));
    skin = imfill(skin, 'holes');

    cc = bwconncomp(skin);
    if cc.NumObjects == 0
        refinedBoxes(i,:) = bbox;      % ten bulunamazsa ilk kutuyu koru
        continue;
    end
    [~, big] = max(cellfun(@numel, cc.PixelIdxList));
    faceMask = false(size(skin));
    faceMask(cc.PixelIdxList{big}) = true;
    bb = regionprops(faceMask, 'BoundingBox').BoundingBox;
    refinedBoxes(i,:) = [rx1 + bb(1) - 0.5, ry1 + bb(2) - 0.5, bb(3), bb(4)];
end

% Cene golgede kaldigi icin alt kenari, her yuzun kendi yuksekligine gore uzat
faceH = double(faces(:,4));
refinedBoxes(:,4) = min(refinedBoxes(:,4) + 0.12*faceH, ...
    size(img,1) - refinedBoxes(:,2));

% 3) Gosterim
imgOriginalBox = insertShape(img, 'Rectangle', rawFaces, 'Color', 'red', 'LineWidth', 4);
imgRefinedBox  = insertShape(img, 'Rectangle', refinedBoxes, 'Color', 'green', 'LineWidth', 4);
imgBoth = insertShape(imgOriginalBox, 'Rectangle', refinedBoxes, 'Color', 'green', 'LineWidth', 4);

figure('Position', [100 100 1500 450]);
subplot(1,3,1); imshow(imgOriginalBox); title('Viola-Jones bounding box');
subplot(1,3,2); imshow(imgRefinedBox);  title('Dogrulanmis ve iyilestirilmis yuzler');
subplot(1,3,3); imshow(imgBoth);        title('Kirmizi: Viola-Jones, Yesil: Iyilestirilmis');

fprintf('Ham Viola-Jones adayi: %d\n', size(rawFaces,1));
fprintf('Dogrulanan yuz       : %d\n', size(faces,1));

% ---------------- Yardimci fonksiyonlar ----------------

function valid = validateFace(rgb,gray,box,eyeBig,eyeSmall,nose)
    imageSize = size(gray);
    x = box(1); y = box(2); w = box(3); h = box(4);
    eyeROI = clipROI([x,y+0.08*h,w,0.55*h],imageSize);
    noseROI = clipROI([x+0.15*w,y+0.30*h,0.70*w,0.58*h],imageSize);

    eyes = zeros(0,4);
    if eyeROI(3) >= 45 && eyeROI(4) >= 11
        eyes = eyeBig(gray,eyeROI);
    end
    if isempty(eyes) && eyeROI(3) >= 22 && eyeROI(4) >= 5
        eyes = eyeSmall(gray,eyeROI);
    end

    noses = zeros(0,4);
    if noseROI(3) >= 18 && noseROI(4) >= 15
        noses = nose(gray,noseROI);
    end

    faceROI = clipROI(box,imageSize);
    x1 = faceROI(1); y1 = faceROI(2);
    crop = rgb(y1:y1+faceROI(4)-1, x1:x1+faceROI(3)-1, :);
    color = rgb2ycbcr(crop);
    Cb = color(:,:,2); Cr = color(:,:,3);
    skinRatio = nnz((Cb>=77 & Cb<=127) & (Cr>=133 & Cr<=173)) / ...
        (size(Cb,1)*size(Cb,2));

    score = 2*~isempty(eyes) + ~isempty(noses) + ...
        (skinRatio >= 0.12 && skinRatio <= 0.85);
    valid = score >= 3;
end


function roi = clipROI(box,imageSize)
    x1 = max(1,floor(box(1))); y1 = max(1,floor(box(2)));
    x2 = min(imageSize(2),ceil(box(1)+box(3)-1));
    y2 = min(imageSize(1),ceil(box(2)+box(4)-1));
    roi = [x1,y1,max(1,x2-x1+1),max(1,y2-y1+1)];
end
