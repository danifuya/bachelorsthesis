clear all;

imagefiles = dir('*.png');      
nfiles = length(imagefiles);    % Number of files found
count=0;
for i=1:nfiles
   currentfilename = imagefiles(i).name;
   currentimage = imread(currentfilename);
   [width,height,depth]=size(currentimage);
   
   if(width<180 || height<180)
    count=count+1;
   end  
   %name to JPEG
%    newname=erase(currentfilename,"jpg");
%    newname=newname+"jpeg";
%    imwrite(currentimage,Folder+"\"+newSubFolder+"\"+newname,'Quality',Q);
end

%j=imcrop(i);
