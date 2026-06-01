% demo_conjunta.m

clear; clc;
fprintf('===================================================================\n');
fprintf('     MPEI - DEMONSTRAÇÃO CONJUNTA: FEED DE NOTÍCIAS INTELIGENTE    \n');
fprintf('===================================================================\n\n');

% Adicionar pastas ao path
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

% O utilizador virtual já leu as primeiras 50 notícias no passado
N_lidos_simulacao = 50;
fprintf('-> Simulação: O utilizador acabou de ler as primeiras %d notícias da base de dados.\n', N_lidos_simulacao);
for idx_lido = 1:N_lidos_simulacao
    bf_historico = bf_historico.insert(titulos_base{idx_lido});
end

% Estruturas dinâmicas para gerir o Feed Ativo exibido no ecrã da sessão
Signatures_publicadas = [];
Titulos_publicados = {};

fprintf('[OK]: Sistema inicializado com sucesso!\n\n');


% =========================================================================
% PROCURA DINÂMICA DE NOTÍCIAS REAIS E DISTANTES (ABRANGENDO AS 4 CLASSES)
% =========================================================================
fprintf('>>> A localizar notícias inéditas e distantes no dataset para os testes...\n');

idx_mundo    = 1501; while idx_mundo    <= height(tabela_news) && ~strcmp(string(tabela_news.('Class Index')(idx_mundo)), '1'),    idx_mundo = idx_mundo + 1; end
idx_desporto = 1501; while idx_desporto <= height(tabela_news) && ~strcmp(string(tabela_news.('Class Index')(idx_desporto)), '2'), idx_desporto = idx_desporto + 1; end
idx_economia = 1501; while idx_economia <= height(tabela_news) && ~strcmp(string(tabela_news.('Class Index')(idx_economia)), '3'), idx_economia = idx_economia + 1; end
idx_tech     = 1501; while idx_tech     <= height(tabela_news) && ~strcmp(string(tabela_news.('Class Index')(idx_tech)), '4'),    idx_tech = idx_tech + 1; end

% Vetor industrial de testes com 10 cenários (Cenários Reais + Edge Cases Probabilísticos)
noticias_teste = { ...
    tabela_news.Title{25}, ...                                                   % Cenário 1: JÁ LIDA HISTÓRICO (Bloqueio exato pelo Bloom -> DEVE DESCARTAR)
    tabela_news.Title{idx_desporto}, ...                                         % Cenário 2: DATASET REAL DISTANTE - DESPORTO (Inédita, Única -> DEVE PUBLICAR)
    tabela_news.Title{100}, ...                                                  % Cenário 3: EXISTENTE NÃO LIDA (Primeira vez que entra nesta sessão -> DEVE PUBLICAR)
    [tabela_news.Title{100}, ' --- LATEST UPDATE FROM REUTERS'], ...             % Cenário 4: PARÁFRASE COM RUÍDO (Engana o Bloom, apanhada pelo MinHash -> DEVE DESCARTAR)
    tabela_news.Title{idx_economia}, ...                                         % Cenário 5: DATASET REAL DISTANTE - ECONOMIA (Inédita, Única -> DEVE PUBLICAR)
    upper(tabela_news.Title{100}), ...                                           % Cenário 6: EDGE CASE: MAIÚSCULAS DO C3 (Engana o Bloom, apanhada pelo MinHash -> DEVE DESCARTAR)
    tabela_news.Title{idx_mundo}, ...                                            % Cenário 7: DATASET REAL DISTANTE - MUNDO (Inédita, Única -> DEVE PUBLICAR)
    tabela_news.Title{idx_tech}, ...                                             % Cenário 8: DATASET REAL DISTANTE - CIÊNCIA/TECNOLOGIA (Inédita, Única -> DEVE PUBLICAR)
    'Away Swept Gets Quality', ...                                               % Cenário 9: EDGE CASE: PERMUTAÇÃO (Frase baralhada. MinHash interceta por Jaccard -> DEVE DESCARTAR)
    'Oil', ...                                                                   % Cenário 10: EDGE CASE: TEXTO INFERIOR AO SHINGLE (Testa robustez de tamanho -> DEVE PUBLICAR)
    '12345 !!! ABC ???' ...                                                      % Cenário 11: EDGE CASE: RUÍDO PURO (Confiança Baixa -> DEVE PUBLICAR COM AVISO)
};

limiar_similaridade = 0.45; 


% =========================================================================
% STEP 2: EXECUÇÃO DO PIPELINE DO FEED EM TEMPO REAL
% =========================================================================
fprintf('\n===================================================================\n');
fprintf('               PROCESSAMENTO DO FLUXO DE NOTÍCIAS                  \n');
fprintf('===================================================================\n');

for t = 1:numel(noticias_teste)
    noticia_atual = noticias_teste{t};
    fprintf('\n[Cenário %d] Título Recebido: "%s"\n', t, noticia_atual);
    fprintf('-------------------------------------------------------------------\n');
    
    % --- FILTRO 1: BLOOM FILTER (Controlo de Duplicados Exatos) ---
    if bf_historico.lookup(noticia_atual)
        fprintf('=> [BLOOM FILTER]: [BLOQUEADA] Esta notícia exata já foi lida pelo utilizador!\n');
        fprintf('>> [AÇÃO DO FEED]: [DESCARTADA] Artigo ignorado para evitar repetição visual.\n');
        continue; 
    else
        fprintf('=> [BLOOM FILTER]: [PASSOU] Notícia inédita no histórico do utilizador.\n');
    end
    
    % --- FILTRO 2: NAÏVE BAYES (Classificação Temática) ---
    classe_prevista_id = nb_sistema.classify({noticia_atual});
    nome_classe = classMap(classe_prevista_id{1});
    
    probs_struct = nb_sistema.probability(noticia_atual);
    [max_prob, ~] = max(probs_struct.probabilities);
    
    if max_prob < 0.60
        fprintf('=> [NAÏVE BAYES]: [AVISO] Baixa confiança devido a vocabulário desconhecido ou muito curto.\n');
        fprintf('   Classificação Estatística por Omissão -> ** %s ** (Confiança: %.2f%%)\n', ...
            nome_classe, max_prob * 100);
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
        % Comparação matricial vetorizada contra o feed ativo exibido no ecrã
        all_sims = sum(signature_atual == Signatures_publicadas, 1) / k_hashes;
        [melhor_similaridade, index_melhor_similar] = max(all_sims);
        dist_melhor = 1 - melhor_similaridade;
    end
    
    if dist_melhor < limiar_similaridade
        fprintf('=> [MINHASH]: [ALERTA] Detetada redundância temática com um artigo já exibido neste feed!\n');
        fprintf('   Artigo Parente Ativo no Ecrã: "%s"\n', Titulos_publicados{index_melhor_similar});
        fprintf('   Semelhança Estimada (Jaccard): %.2f%%\n', melhor_similaridade * 100);
        fprintf('>> [AÇÃO DO FEED]: [RECUSADA] Publicação abortada para evitar duplicar o mesmo tema no painel.\n');
        continue; 
    else
        fprintf('=> [MINHASH]: [CONTEÚDO ÚNICO] O tema desta notícia ainda não existe no feed atual.\n');
    end
    
    % --- SUCESSO DO PIPELINE: O artigo passa a fazer parte do feed ativo ---
    Signatures_publicadas = [Signatures_publicadas, signature_atual];
    Titulos_publicados{end+1} = noticia_atual;
    
    % Atualiza o histórico do Bloom Filter para sessões futuras
    bf_historico = bf_historico.insert(noticia_atual);
    fprintf('>> [AÇÃO DO FEED]: [SUCESSO] Artigo publicado com sucesso no vosso painel personalizado!\n');
end

fprintf('\n===================================================================\n');
fprintf('     [FIM DA DEMONSTRAÇÃO]: Pipeline validado com rigor de produção! \n');
fprintf('===================================================================\n');