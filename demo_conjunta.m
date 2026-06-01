% demo_conjunta.m
clear; clc; close all;
fprintf('===================================================================\n');
fprintf('     MPEI - DEMONSTRAÇÃO CONJUNTA: FEED DE NOTÍCIAS INTELIGENTE    \n');
fprintf('===================================================================\n\n');

% Adicionar pastas ao path para garantir acesso aos ficheiros de classes (.m)
addpath(genpath(pwd));

% =========================================================================
% STEP 1: CARREGAMENTO DOS DADOS E TREINO DO SISTEMA
% =========================================================================
fprintf('>>> [Fase de Inicialização] A carregar a base de dados de notícias...\n');
opts = detectImportOptions('news.csv', 'VariableNamingRule', 'preserve');
tabela_news = readtable('news.csv', opts);

% Normalização defensiva do tipo de dados para garantir compatibilidade
tabela_news.Title = cellstr(tabela_news.Title);

% Mapeamento dos índices de classe textuais para abranger as 4 categorias nativas do AG News
classMap = containers.Map({'1','2','3','4'}, {'Mundo', 'Desporto', 'Economia', 'Ciência/Tecnologia'});

% Usar os primeiros 1500 registos para alimentar o conhecimento do sistema
N_base = min(1500, height(tabela_news));
titulos_base = tabela_news.Title(1:N_base);
classes_base = tabela_news.('Class Index')(1:N_base);
if isnumeric(classes_base), classes_base = cellstr(string(classes_base)); end

fprintf('A treinar o classificador Naïve Bayes com %d notícias...\n', N_base);
nb_sistema = naiveBayes('multinomial');
nb_sistema = nb_sistema.train(titulos_base, classes_base);

fprintf('A inicializar o gerador MinHash e o histórico do utilizador...\n');
k_hashes = 150; 
mh_sistema = minHash(k_hashes);

bf_historico = bloomFilter(500, 0.01, 'classic');

% O utilizador virtual já leu as primeiras 50 notícias no passado (Filtro de Histórico)
N_lidos_simulacao = 50;
fprintf('-> Simulação: O utilizador acabou de ler as primeiras %d notícias da base de dados.\n', N_lidos_simulacao);
for idx_lido = 1:N_lidos_simulacao
    bf_historico = bf_historico.insert(titulos_base{idx_lido});
end

% Estruturas dinâmicas para gerir o Feed Ativo exibido no ecrã da sessão
Signatures_publicadas = [];
Titulos_publicados = {};

% Repositórios categorizados para o Painel Consolidado do utilizador
feed_categorias = struct('Mundo', {{}}, 'Desporto', {{}}, 'Economia', {{}}, 'Ciencia', {{}});

% Telemetria e Dashboard de Análise Estatística
stats = struct('total', 0, 'bloom_bloq', 0, 'minhash_bloq', 0, 'publicados', 0);
fprintf('[OK]: Sistema inicializado com sucesso!\n\n');


% =========================================================================
% EXTRAÇÃO DINÂMICA DE EXEMPLOS DISTANTES DO DATASET (SEM COISAS HARDCODED)
% =========================================================================
fprintf('>>> A localizar notícias inéditas e distantes no dataset para os testes...\n');

% Procurar o primeiro bloco de notícias inéditas após a linha 1500
idx_mundo    = 1501; while idx_mundo    <= height(tabela_news) && ~strcmp(string(tabela_news.('Class Index')(idx_mundo)), '1'),    idx_mundo = idx_mundo + 1; end
idx_desporto = 1501; while idx_desporto <= height(tabela_news) && ~strcmp(string(tabela_news.('Class Index')(idx_desporto)), '2'), idx_desporto = idx_desporto + 1; end
idx_economia = 1501; while idx_economia <= height(tabela_news) && ~strcmp(string(tabela_news.('Class Index')(idx_economia)), '3'), idx_economia = idx_economia + 1; end
idx_tech     = 1501; while idx_tech     <= height(tabela_news) && ~strcmp(string(tabela_news.('Class Index')(idx_tech)), '4'),    idx_tech = idx_tech + 1; end

% Procurar um segundo bloco de notícias mais distantes para diversificar o feed publicado
idx_mundo2    = idx_mundo + 1;    while idx_mundo2    <= height(tabela_news) && ~strcmp(string(tabela_news.('Class Index')(idx_mundo2)), '1'),    idx_mundo2 = idx_mundo2 + 1; end
idx_desporto2 = idx_desporto + 1; while idx_desporto2 <= height(tabela_news) && ~strcmp(string(tabela_news.('Class Index')(idx_desporto2)), '2'), idx_desporto2 = idx_desporto2 + 1; end
idx_economia2 = idx_economia + 1; while idx_economia2 <= height(tabela_news) && ~strcmp(string(tabela_news.('Class Index')(idx_economia2)), '3'), idx_economia2 = idx_economia2 + 1; end
idx_tech2     = idx_tech + 1;     while idx_tech2     <= height(tabela_news) && ~strcmp(string(tabela_news.('Class Index')(idx_tech2)), '4'),    idx_tech2 = idx_tech2 + 1; end

% MONTAGEM DO VETOR INDUSTRIAL DE 15 CENÁRIOS COM MANIPULAÇÃO PROGRAMÁTICA
noticias_teste = { ...
    tabela_news.Title{15}, ...                                                   % C1: JÁ LIDA HISTÓRICO (Bloqueio exato pelo Bloom)
    tabela_news.Title{42}, ...                                                   % C2: JÁ LIDA HISTÓRICO (Bloqueio exato pelo Bloom)
    tabela_news.Title{idx_mundo}, ...                                            % C3: DATASET REAL - MUNDO (Deve Publicar)
    tabela_news.Title{idx_desporto}, ...                                         % C4: DATASET REAL - DESPORTO (Deve Publicar)
    tabela_news.Title{idx_economia}, ...                                         % C5: DATASET REAL - ECONOMIA (Deve Publicar)
    tabela_news.Title{idx_tech}, ...                                             % C6: DATASET REAL - CIÊNCIA/TECH (Deve Publicar)
    tabela_news.Title{120}, ...                                                  % C7: EXISTENTE NÃO LIDA (Primeira vez na sessão -> Deve Publicar)
    [tabela_news.Title{120}, ' --- LATEST WIRE VIA REUTERS'], ...                % C8: PARÁFRASE COM RUÍDO DO C7 (Apanhada pelo MinHash -> Deve Rejeitar)
    upper(tabela_news.Title{120}), ...                                           % C9: EDGE CASE: MAIÚSCULAS DO C7 (Apanhada pelo MinHash -> Deve Rejeitar)
    tabela_news.Title{250}, ...                                                  % C10: EXISTENTE NÃO LIDA (Primeira vez na sessão -> Deve Publicar)
    strjoin(flip(strsplit(tabela_news.Title{250})), ' '), ...                    % C11: EDGE CASE: PERMUTAÇÃO DE PALAVRAS DO C10 (MinHash apanha -> Deve Rejeitar)
    strtok(tabela_news.Title{400}), ...                                          % C12 CORRIGIDA: Extrai a primeira palavra de forma segura sem quebras de indexação
    tabela_news.Title{idx_mundo2}, ...                                           % C13: DATASET REAL - MUNDO 2 (Deve Publicar)
    tabela_news.Title{idx_desporto2}, ...                                        % C14: DATASET REAL - DESPORTO 2 (Deve Publicar)
    tabela_news.Title{idx_economia2} ...                                         % C15: DATASET REAL - ECONOMIA 2 (Deve Publicar)
};

limiar_similaridade = 0.45; 
% Matriz de telemetria streaming para guardar o estado histórico passo a passo
historico_telemetria = zeros(numel(noticias_teste), 3);


% =========================================================================
% STEP 2: EXECUÇÃO DO PIPELINE DO FEED EM TEMPO REAL
% =========================================================================
fprintf('\n===================================================================\n');
fprintf('               PROCESSAMENTO DO FLUXO DE NOTÍCIAS                  \n');
fprintf('===================================================================\n');

for t = 1:numel(noticias_teste)
    noticia_atual = noticias_teste{t};
    stats.total = stats.total + 1;
    
    fprintf('\n[Cenário %d] Título Recebido: "%s"\n', t, noticia_atual);
    fprintf('-------------------------------------------------------------------\n');
    
    % --- FILTRO 1: BLOOM FILTER (Controlo de Duplicados Exatos) ---
    if bf_historico.lookup(noticia_atual)
        fprintf('=> [BLOOM FILTER]: [BLOQUEADA] Esta notícia exata já foi lida pelo utilizador!\n');
        fprintf('>> [AÇÃO DO FEED]: [DESCARTADA] Artigo ignorado para evitar repetição visual.\n');
        stats.bloom_bloq = stats.bloom_bloq + 1;
        % Guardar histórico corrente de streaming e saltar
        historico_telemetria(t, :) = [stats.bloom_bloq, stats.minhash_bloq, stats.publicados];
        continue; 
    else
        fprintf('=> [BLOOM FILTER]: [PASSOU] Notícia inédita no histórico do utilizador.\n');
    end
    
    % --- FILTRO 2: NAÏVE BAYES (Classificação Temática) ---
    classe_prevista_id = nb_sistema.classify({noticia_atual});
    nome_classe = classMap(classe_prevista_id{1});
    
    probs_struct = nb_sistema.probability(noticia_atual);
    [max_prob, ~] = max(probs_struct.probabilities);
    
    flag_aviso_ruido = false;
    if max_prob < 0.60
        fprintf('=> [NAÏVE BAYES]: [AVISO] Baixa confiança devido a vocabulário desconhecido ou muito curto.\n');
        fprintf('   Classificação Estatística por Omissão -> ** %s ** (Confiança: %.2f%%)\n', ...
            nome_classe, max_prob * 100);
        flag_aviso_ruido = true;
    else
        fprintf('=> [NAÏVE BAYES]: [CATEGORIZADA] Destino do Feed -> ** %s ** (Confiança: %.2f%%)\n', ...
            nome_classe, max_prob * 100);
    end
    
    % --- FILTRO 3: MINHASH (Deteção de Similaridade Semântica no Feed Exibido) ---
    shingles_atual = mh_sistema.createShingles(noticia_atual, 3, 'chars');
    signature_atual = mh_sistema.getSignature(shingles_atual);
    
    if isempty(Signatures_publicadas)
        melhor_similaridade = 0;
        dist_melhor = 1;
    else
        % Comparação matricial vetorizada em bloco contra a sessão ativa
        all_sims = sum(signature_atual == Signatures_publicadas, 1) / k_hashes;
        [melhor_similaridade, index_melhor_similar] = max(all_sims);
        dist_melhor = 1 - melhor_similaridade;
    end
    
    if dist_melhor < limiar_similaridade
        fprintf('=> [MINHASH]: [ALERTA] Detetada redundância temática com um artigo já exibido neste feed!\n');
        fprintf('   Artigo Parente Ativo no Ecrã: "%s"\n', Titulos_publicados{index_melhor_similar});
        fprintf('   Semelhança Estimada (Jaccard): %.2f%%\n', melhor_similaridade * 100);
        fprintf('>> [AÇÃO DO FEED]: [RECUSADA] Publicação abortada para evitar duplicar o mesmo tema no painel.\n');
        stats.minhash_bloq = stats.minhash_bloq + 1;
        historico_telemetria(t, :) = [stats.bloom_bloq, stats.minhash_bloq, stats.publicados];
        continue; 
    else
        fprintf('=> [MINHASH]: [CONTEÚDO ÚNICO] O tema desta notícia ainda não existe no feed atual.\n');
    end
    
    % --- SUCESSO DO PIPELINE: O artigo é aceite pelo ecossistema ---
    Signatures_publicadas = [Signatures_publicadas, signature_atual];
    Titulos_publicados{end+1} = noticia_atual;
    stats.publicados = stats.publicados + 1;
    
    % Atualiza o histórico do Bloom Filter para sessões futuras
    bf_historico = bf_historico.insert(noticia_atual);
    
    % Roteamento dinâmico para a pasta correspondente
    prefixo_exibicao = '';
    if flag_aviso_ruido, prefixo_exibicao = '[AVISO DE CONTEÚDO] '; end
    
    switch classe_prevista_id{1}
        case '1', feed_categorias.Mundo{end+1} = [prefixo_exibicao, noticia_atual];
        case '2', feed_categorias.Desporto{end+1} = [prefixo_exibicao, noticia_atual];
        case '3', feed_categorias.Economia{end+1} = [prefixo_exibicao, noticia_atual];
        case '4', feed_categorias.Ciencia{end+1} = [prefixo_exibicao, noticia_atual];
    end
    fprintf('>> [AÇÃO DO FEED]: [SUCESSO] Artigo encaminhado e publicado na secção de %s!\n', nome_classe);
    
    % Guardar histórico corrente de streaming
    historico_telemetria(t, :) = [stats.bloom_bloq, stats.minhash_bloq, stats.publicados];
end

% =========================================================================
% STEP 3: INTERFACE DE RESULTADOS CONSOLIDADA
% =========================================================================
fprintf('\n===================================================================\n');
fprintf('             PAINEL PERSONALIZADO DE NOTÍCIAS DO UTILIZADOR        \n');
fprintf('===================================================================\n');
categorias_nomes = fieldnames(feed_categorias);
for cat_idx = 1:numel(categorias_nomes)
    cat_nome = categorias_nomes{cat_idx};
    titulos_na_cat = feed_categorias.(cat_nome);
    
    label_interface = cat_nome;
    if strcmp(cat_nome, 'Ciencia'), label_interface = 'Ciência/Tecnologia'; end
    
    fprintf('\n>>> SECÇÃO: %s (%d artigos)\n', upper(label_interface), numel(titulos_na_cat));
    fprintf('-------------------------------------------------------------------\n');
    if isempty(titulos_na_cat)
        fprintf('   [Nenhum artigo novo publicado nesta categoria durante a sessão]\n');
    else
        for t_cat = 1:numel(titulos_na_cat)
            fprintf('   [%d] %s\n', t_cat, titulos_na_cat{t_cat});
        end
    end
end

% Imprimir a Telemetria e Auditoria Estatística da Sessão
fprintf('\n===================================================================\n');
fprintf('                  TELEMETRIA DO STREAMING DE DADOS                 \n');
fprintf('===================================================================\n');
fprintf('Artigos Submetidos ao Pipeline:      %d\n', stats.total);
fprintf('Bloqueados por Histórico (Bloom):    %d (Taxa: %.1f%%)\n', stats.bloom_bloq, (stats.bloom_bloq/stats.total)*100);
fprintf('Recusados por Duplicação (MinHash):  %d (Taxa: %.1f%%)\n', stats.minhash_bloq, (stats.minhash_bloq/stats.total)*100);
fprintf('Publicados com Sucesso no Feed:      %d (Taxa: %.1f%%)\n', stats.publicados, (stats.publicados/stats.total)*100);
fprintf('===================================================================\n');

% =========================================================================
% STEP 4: REPRESENTAÇÃO GRÁFICA AVANÇADA (PIE3 + STACKED BAR TELEMETRY)
% =========================================================================
volumes_finais = [numel(feed_categorias.Mundo), numel(feed_categorias.Desporto), ...
                  numel(feed_categorias.Economia), numel(feed_categorias.Ciencia)];
labels_grafico = {'Mundo', 'Desporto', 'Economia', 'Sci/Tech'};

if sum(volumes_finais) > 0
    % FIGURA 1: Composição Temática (Pie Chart)
    figure(1);
    pie3(volumes_finais, volumes_finais > 0, labels_grafico);
    title('Composição Temática das Notícias Publicadas no Feed Ativo');
    fprintf('\n[GRÁFICO]: Janela Gráfica Figura 1 (Pie3) gerada com sucesso!\n');
    
    % FIGURA 2: Telemetria Streaming Passo a Passo (Stacked Bar)
    figure(2);
    bar(historico_telemetria, 'stacked', 'EdgeColor', 'none');
    grid on;
    xlabel('Passos de Processamento (Streaming Index)');
    ylabel('Contagem Acumulada de Artigos');
    title('Evolução Dinâmica do Pipeline do Feed de Notícias');
    legend({'Bloqueados Bloom (Histórico)', 'Recusados MinHash (Duplicados)', 'Publicados no Painel'}, ...
            'Location', 'NorthWest');
    fprintf('[GRÁFICO]: Janela Gráfica Figura 2 (Stacked Bar Telemetry) gerada com sucesso!\n');
end

fprintf('\n===================================================================\n');
fprintf('     [FIM DA DEMONSTRAÇÃO]: Pipeline unificado operado com distinção! \n');
fprintf('===================================================================\n');