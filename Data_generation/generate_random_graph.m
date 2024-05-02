function G_Adj = generate_random_graph( N, method, m, p )

command = char(strcat('python Data_generation/nx_random_graph.py',{' '},num2str(N),{' '},method,{' '},num2str(m),{' '},num2str(p)));
[~,cmdout] = system(command);
G_Adj = -1;
while 1 == 1
    if strcmp(strcat(cmdout,''),'success') == 1
        G_Adj = load('Data_generation/Adj.csv');
        break;
    end
end

end

