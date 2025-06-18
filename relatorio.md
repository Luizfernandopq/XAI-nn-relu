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

