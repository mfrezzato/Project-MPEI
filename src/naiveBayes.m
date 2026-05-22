function modelo = nb_treinar(documentos, etiquetas, modo)

    if nargin < 3
        modo = 'multinomial';
    end

    fprintf('[NaiveBayes] A treinar modelo (%s) com %d documentos...\n', ...
            modo, numel(documentos));

    classes = unique(etiquetas);
    num_classes = numel(classes);

    vocabulario = construir_vocabulario(documentos);
    V = numel(vocabulario);
    fprintf('[NaiveBayes] Vocabulário: %d palavras únicas\n', V);

    log_prior = zeros(1, num_classes);
    for i = 1:num_classes
        count = sum(strcmp(etiquetas, classes{i}));
        log_prior(i) = log(count / numel(etiquetas));
    end

    log_likelihood = zeros(num_classes, V);

    for i = 1:num_classes
        idx_classe = strcmp(etiquetas, classes{i});
        docs_classe = documentos(idx_classe);

        if strcmp(modo, 'multinomial')
            contagens = zeros(1, V);
            for d = 1:numel(docs_classe)
                palavras = tokenizar(docs_classe{d});
                for p = 1:numel(palavras)
                    idx_palavra = find(strcmp(vocabulario, palavras{p}), 1);
                    if ~isempty(idx_palavra)
                        contagens(idx_palavra) = contagens(idx_palavra) + 1;
                    end
                end
            end
            total = sum(contagens);
            log_likelihood(i, :) = log((contagens + 1) / (total + V));

        elseif strcmp(modo, 'bernoulli')
            contagens = zeros(1, V);
            N_classe = numel(docs_classe);
            for d = 1:N_classe
                palavras_unicas = unique(tokenizar(docs_classe{d}));
                for p = 1:numel(palavras_unicas)
                    idx_palavra = find(strcmp(vocabulario, palavras_unicas{p}), 1);
                    if ~isempty(idx_palavra)
                        contagens(idx_palavra) = contagens(idx_palavra) + 1;
                    end
                end
            end
            log_likelihood(i, :) = log((contagens + 1) / (N_classe + 2));
        end
    end

    modelo.classes        = classes;
    modelo.vocabulario    = vocabulario;
    modelo.log_prior      = log_prior;
    modelo.log_likelihood = log_likelihood;
    modelo.modo           = modo;
    modelo.num_docs_treino = numel(documentos);
    modelo.V              = V;

    fprintf('[NaiveBayes] Treino concluído. Classes: %s\n', ...
            strjoin(classes, ', '));
end

function predicoes = nb_classificar(modelo, documentos)

    if ischar(documentos)
        documentos = {documentos};
    end

    num_docs = numel(documentos);
    predicoes = cell(1, num_docs);

    for d = 1:num_docs
        [classe, ~] = nb_classificar_um(modelo, documentos{d});
        predicoes{d} = classe;
    end
end

function probs = nb_probabilidade(modelo, documento)

    [~, log_scores] = nb_classificar_um(modelo, documento);

    log_scores = log_scores - max(log_scores);   
    scores_exp = exp(log_scores);
    probabilidades = scores_exp / sum(scores_exp);

    probs.classes        = modelo.classes;
    probs.probabilidades = probabilidades;

    fprintf('\n[NaiveBayes] Probabilidades para: "%s"\n', documento);
    fprintf('  %-20s  %s\n', 'Classe', 'Probabilidade');
    fprintf('  %s\n', repmat('-', 1, 35));
    for i = 1:numel(modelo.classes)
        fprintf('  %-20s  %.4f (%.1f%%)\n', ...
                modelo.classes{i}, probabilidades(i), probabilidades(i)*100);
    end
end

function stats = nb_avaliar(modelo, documentos, etiquetas)

    predicoes = nb_classificar(modelo, documentos);
    classes   = modelo.classes;
    N         = numel(etiquetas);

    corretos = sum(strcmp(predicoes, etiquetas));
    accuracy = corretos / N;

    fprintf('\n[NaiveBayes] === Avaliação do Modelo ===\n');
    fprintf('  Documentos testados : %d\n', N);
    fprintf('  Corretos            : %d\n', corretos);
    fprintf('  Accuracy            : %.4f (%.1f%%)\n', accuracy, accuracy*100);

    fprintf('\n  %-15s  %9s  %9s  %9s\n', 'Classe', 'Precisão', 'Recall', 'F1');
    fprintf('  %s\n', repmat('-', 1, 50));

    precisao_vec = zeros(1, numel(classes));
    recall_vec   = zeros(1, numel(classes));
    f1_vec       = zeros(1, numel(classes));

    for i = 1:numel(classes)
        c = classes{i};
        TP = sum(strcmp(predicoes, c) & strcmp(etiquetas, c));
        FP = sum(strcmp(predicoes, c) & ~strcmp(etiquetas, c));
        FN = sum(~strcmp(predicoes, c) & strcmp(etiquetas, c));

        p  = TP / max(TP + FP, 1);
        r  = TP / max(TP + FN, 1);
        f1 = 2 * p * r / max(p + r, 1e-10);

        precisao_vec(i) = p;
        recall_vec(i)   = r;
        f1_vec(i)       = f1;

        fprintf('  %-15s  %9.4f  %9.4f  %9.4f\n', c, p, r, f1);
    end

    fprintf('  %s\n', repmat('-', 1, 50));
    fprintf('  %-15s  %9.4f  %9.4f  %9.4f\n', 'Média Macro', ...
            mean(precisao_vec), mean(recall_vec), mean(f1_vec));

    fprintf('\n  Matriz de Confusão (linhas=real, colunas=previsto):\n');
    fprintf('  %-12s', '');
    for i = 1:numel(classes)
        fprintf('  %-10s', classes{i});
    end
    fprintf('\n');
    for i = 1:numel(classes)
        fprintf('  %-12s', classes{i});
        for j = 1:numel(classes)
            val = sum(strcmp(etiquetas, classes{i}) & strcmp(predicoes, classes{j}));
            fprintf('  %-10d', val);
        end
        fprintf('\n');
    end

    stats.accuracy  = accuracy;
    stats.precisao  = precisao_vec;
    stats.recall    = recall_vec;
    stats.f1        = f1_vec;
    stats.classes   = classes;
    stats.predicoes = predicoes;
end

function [melhor_classe, log_scores] = nb_classificar_um(modelo, documento)

    palavras = tokenizar(documento);
    num_classes = numel(modelo.classes);
    log_scores  = modelo.log_prior;   

    if strcmp(modelo.modo, 'multinomial')
        for p = 1:numel(palavras)
            idx = find(strcmp(modelo.vocabulario, palavras{p}), 1);
            if ~isempty(idx)
                log_scores = log_scores + modelo.log_likelihood(:, idx)';
            end
        end

    elseif strcmp(modelo.modo, 'bernoulli')
        palavras_unicas = unique(palavras);
        for v = 1:modelo.V
            palavra_v = modelo.vocabulario{v};
            presente  = any(strcmp(palavras_unicas, palavra_v));
            if presente
                log_scores = log_scores + modelo.log_likelihood(:, v)';
            else
                log_scores = log_scores + log(1 - exp(modelo.log_likelihood(:, v)))';
            end
        end
    end

    [~, idx_melhor] = max(log_scores);
    melhor_classe   = modelo.classes{idx_melhor};
end

function palavras = tokenizar(texto)
    texto = lower(texto);
    texto = regexprep(texto, '[^a-záàâãéèêíïóôõúüç0-9\s]', ' ');
    palavras = strsplit(strtrim(texto));
    palavras = palavras(~cellfun(@isempty, palavras));
    stop_words = {'a','o','e','de','do','da','em','um','uma','os','as', ...
                  'que','se','com','por','para','no','na','nos','nas', ...
                  'the','is','in','it','of','and','to','a','an','this'};
    palavras = palavras(~ismember(palavras, stop_words));
end

function vocabulario = construir_vocabulario(documentos)
    todas_palavras = {};
    for i = 1:numel(documentos)
        palavras = tokenizar(documentos{i});
        todas_palavras = [todas_palavras, palavras]; 
    end
    vocabulario = unique(todas_palavras);
    vocabulario = sort(vocabulario);  
end