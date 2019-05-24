clear all;

%Quality
Q=10;
originalformat="bmp"
carpetaActual="groundtruth\all"
Folder="C:\Users\danif\Documents\Eng. telecos\tfg\data\images\test"
%%
%Crear folder para la Q


[parentFolder deepestFolder] = fileparts(Folder);
% Next, create a name for a subfolder within that.
% For example 'D:\photos\Info\DATA-Info'
newSubFolder = sprintf('compressed_Q'+convertCharsToStrings(num2str(Q)),Folder, deepestFolder);
% Finally, create the folder if it doesn't exist already.
if ~exist(newSubFolder, 'dir')
  mkdir(newSubFolder);
end 
%% Read images of current file
%tienes que estar dentro del directorio test, val or train
imagefiles = dir(Folder+"\"+carpetaActual+'\*.'+originalformat);      
nfiles = length(imagefiles);    % Number of files found
for i=1:nfiles
   currentfilename = imagefiles(i).name;
   currentimage = imread(currentfilename);
   %name to JPEG
   newname=erase(currentfilename,originalformat);
   newname=newname+"jpeg";
   imwrite(currentimage,Folder+"\"+newSubFolder+"\"+newname,'Quality',Q);
end

