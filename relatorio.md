# Codificação

Uma codificação padrão deve encontrar bounds para cada neurônio
 - A explicação utiliza os bounds do neurônio (y) sem a incidência da ReLU

## 30/12/24
- Me recordo de que se utilizar ub<0 provoca erros na codificação
- Devo utilizar uma função ReLU(ub) em Explanator?
- Para explicar uma camada intermediária, devo utilizar para y bounds com a ReLU aplicada?
- Resposta: Usar ub<0 causa unsolve (não muda a classe) pois para a rede aquele neurônio é sempre 0, ub deve ser igual a 0
- Nota: as segundas tentativas efetivamente salvam a explicação, o epsilon não
- Nota2: as segundas tentativas aconcetem entre 5% e 30% dos dados do dataset wine e005
- Dúvida: Será que o epsilon diminui a quantidade de segundas tentativas? 

### Verificar:
 - assert está fazendo seu trabalho corretamente?
 - ajustar os bounds, está retornando tudo errado
 - o certo deve ser lb<0 e ub<0 se possível
 - ub<=0 -> return bounds.lb, bounds.ub
 - ub> 0 -> then
 - lb'==0 -> return bounds.lb, ub'
 - lb'> 0 -> return lb', ub'

## 01/01/2025

Resolvido a questão de segundas tentativas (devolver a restrição) com os seguintes passos:

- Inverter a ordem dos outputs contraints (Oi <= Oj). Isso funciona pois o cplex possui uma margem que está associada ao lado direito da equação
- Utilizar uma pequena soma aos outputs (Oi <= Oj + 0.0001).
- Utilizar o epsilon de 0.01 para valores econtrados para bounds
- As segundas tentativas irão aparecer de 0 a 5 vezes no dataset wine

Resultados finais do dia:

- Modifiquei a precisão do modelo para 1e-9
- Se o ub <= t * 1.0001, devolve a restrição
- Se o lb >= t * 0.9999, devolve a restrição
- Essas modificações aparentemente expurgaram os problemas de mudar a classe (Sem precisar adicionanr 0.00001 aos outputs)

## 03/01/2025

Uma explicação de um bound inicial que fure os bounds a frente sempre será o próprio valor do neurônio.
Isso ocorre porque o valor do neurônio acerta em um dos neurônios da próxima camada que foi devolvido a restrição

## 07/03/2025

Irei refatorar meus testes pois irei adicionar novas camadas de verificação

## 09/03/2025

O código legacy foi testado para verificar-se o tamanho das explicações
Foi utilizado os dados de teste em D-Explanations/wine/legacy_02.py com um modelo de 100.00% acurácia

Os resultados foram: 
- Média: 6.491525423728813
- Mediana: 6.0
- Máximo: 12
- Mínimo: 2

Outra rodada, para todo o dataset:

- Média: 6.751054852320675
- Mediana: 6.0
- Máximo: 13
- Mínimo: 3


Formato de explicação deixado para trás para fins de possível utilização

A explicação consiste em um epsilon 0.0 em todos as camadas, devolver a feature todas as vezes
independentemente se ela é importante ou não, para abrir espaços para as outras.
A explicação nesse caso torna-se uma verificação de quais features puderam ser alargadas até o máximo
Esse alargamento consiste em tentar fazer: para neurônio da camada intermediária seguinte, tentar alargar
o máximo da feature através do ub' e do lb'. Salienta-se que não possui garantia, mas tem bom desempenho no mnist.

[59, 167, 423, 429, 434, 465, 677, 911, 959, 971, 1197, 1294, 1316, 1399, 1476, 1538, 1572, 1637, 1772, 1811, 1831, 1932, 1988, 2045, 2214, 2246, 2643, 2668, 2691, 2783, 3006, 3023, 3167, 3268, 3585, 3661, 3667, 4064, 4198, 4275, 4557, 4679, 4855, 4907, 5047, 5192, 5270, 5334, 5349, 5394, 5420, 5431, 5461, 5543, 5637, 5785, 5867, 6009, 6028, 6064, 6225, 6368, 6516, 6526, 6646, 6648, 6747, 6777, 6781, 6804, 6828, 6969, 7051, 7172, 7183, 7281, 7309, 7574, 7610, 7765, 8076, 8093, 8257, 8327, 8366, 8498, 8697, 8740, 8829, 8860, 8875, 9008, 9034, 9280, 9525, 9804, 9837, 9852, 9944, 9998]
Rodando: Net_2x16_hidden relax: 0
Explicação iniciada após: 12.891282796859741
Checkpoint Explicado 1: 19.40043879998848 | média: 19.396409899927676 | Fidelidade 1.0
Checkpoint Explicado 21: 16.893262699712068 | média: 33.23096832850327 | Fidelidade 0.9523809523809523
Checkpoint Explicado 41: 26.165952499955893 | média: 28.88571865609052 | Fidelidade 0.926829268292683
Checkpoint Explicado 61: 18.335331600159407 | média: 31.11415790656551 | Fidelidade 0.9344262295081968
Checkpoint Explicado 81: 125.62070789979771 | média: 36.86310364074867 | Fidelidade 0.9259259259259259
testar digits com 2 x 48 instância 804