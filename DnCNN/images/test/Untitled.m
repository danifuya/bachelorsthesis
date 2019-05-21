imagefiles = dir("*.bmp");     
nfiles = length(imagefiles);    % Number of files found
for i=1:nfiles
   currentfilename = imagefiles(i).name;
   images{i,1}=currentfilename;
   images{i,2}= imread(currentfilename);
end

%%
imagefiles = dir("*.jpeg");      
nfiles = length(imagefiles);    % Number of files found
for i=1:nfiles
   currentfilename = imagefiles(i).name;
   images{i,1}=currentfilename;
   images{i,2}= imread(currentfilename);
end
%%
for i=1:nfiles
    psnrims{1,i}=images_c{i,1};
    psnrims{2,i}=psnr(images_c{i,2},images{i,2},255);
end