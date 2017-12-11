X=[];
y=[];
X = [X,extract_image('85106.gif')]; y = [y,[8 5 1 0 6]];
X = [X,extract_image('72102.gif')]; y = [y,[7 2 1 0 2]];
X = [X,extract_image('58682.gif')]; y = [y,[5 8 6 8 2]];
X = [X,extract_image('57433.gif')]; y = [y,[5 7 4 3 3]];
X = [X,extract_image('32783.gif')]; y = [y,[3 2 7 8 3]];
X = [X,extract_image('30575.gif')]; y = [y,[3 0 5 7 5]];
X = [X,extract_image('25453.gif')]; y = [y,[2 5 4 5 3]];
X = [X,extract_image('22856.gif')]; y = [y,[2 2 8 5 6]];
X = [X,extract_image('17355.gif')]; y = [y,[1 7 3 5 5]];
X = [X,extract_image('05681.gif')]; y = [y,[0 5 6 8 1]];
save ./knn/hack_data.mat X y