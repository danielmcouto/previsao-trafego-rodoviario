# Estrutura mínima dos dados

A base consolidada deve conter:

- `analisador`: identificador do equipamento;
- `faixa`: faixa de rolamento;
- `data_hora`: instante da observação;
- `passeio`: volume de veículos de passeio;
- `comercial`: volume de veículos comerciais;
- `dado_projetado`: 0 para observado e 1 para preenchido/projetado.

Para reproduzir os Modelos B e C, inclua também o calendário especial e as variáveis territoriais utilizadas no estudo. Os nomes podem ser ajustados nas listas localizadas no início do script.
