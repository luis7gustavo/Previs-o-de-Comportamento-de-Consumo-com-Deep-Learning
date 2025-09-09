CREATE DATABASE IF NOT EXISTS bd_comprebem01;
USE bd_comprebem01;
SET FOREIGN_KEY_CHECKS=1;

CREATE TABLE Categorias (
    CategoriaID INT AUTO_INCREMENT PRIMARY KEY,
    NomeCategoria VARCHAR(100) NOT NULL,
    Descricao TEXT
);

CREATE TABLE Clientes (
    ClienteID INT AUTO_INCREMENT PRIMARY KEY,
    Nome VARCHAR(100) NOT NULL,
    Email VARCHAR(100) NOT NULL UNIQUE,
    SenhaHash VARCHAR(255) NOT NULL,
    Endereco VARCHAR(255),
    Cidade VARCHAR(100),
    Estado CHAR(2),
    CEP VARCHAR(9),
    Telefone VARCHAR(20),
    DataCadastro DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE Produtos (
    ProdutoID INT AUTO_INCREMENT PRIMARY KEY,
    Nome VARCHAR(150) NOT NULL,
    Descricao TEXT,
    Preco DECIMAL(10, 2) NOT NULL,
    CategoriaID INT,
    QuantidadeEstoque INT NOT NULL DEFAULT 0,
    AtributosEspecificos JSON,
    CONSTRAINT fk_produto_categoria
        FOREIGN KEY (CategoriaID)
        REFERENCES Categorias(CategoriaID)
        ON DELETE SET NULL
        ON UPDATE CASCADE
);

CREATE TABLE Pedidos (
    PedidoID INT AUTO_INCREMENT PRIMARY KEY,
    ClienteID INT NOT NULL,
    DataPedido DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    StatusPedido VARCHAR(50) NOT NULL,
    ValorTotal DECIMAL(10, 2) NOT NULL,
    EnderecoEntrega VARCHAR(255) NOT NULL,
    CONSTRAINT fk_pedido_cliente
        FOREIGN KEY (ClienteID)
        REFERENCES Clientes(ClienteID)
        ON DELETE RESTRICT
        ON UPDATE CASCADE
);

CREATE TABLE Itens_Pedido (
    ItemPedidoID INT AUTO_INCREMENT PRIMARY KEY,
    PedidoID INT NOT NULL,
    ProdutoID INT NOT NULL,
    Quantidade INT NOT NULL,
    PrecoUnitario DECIMAL(10, 2) NOT NULL,
    CONSTRAINT fk_item_pedido
        FOREIGN KEY (PedidoID)
        REFERENCES Pedidos(PedidoID)
        ON DELETE CASCADE,
    CONSTRAINT fk_item_produto
        FOREIGN KEY (ProdutoID)
        REFERENCES Produtos(ProdutoID)
        ON DELETE RESTRICT
);

CREATE TABLE Pagamentos (
    PagamentoID INT AUTO_INCREMENT PRIMARY KEY,
    PedidoID INT NOT NULL,
    MetodoPagamento VARCHAR(50) NOT NULL,
    Valor DECIMAL(10, 2) NOT NULL,
    DataPagamento DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    StatusPagamento VARCHAR(50) NOT NULL,
    IDTransacaoGateway VARCHAR(100),
    CONSTRAINT fk_pagamento_pedido
        FOREIGN KEY (PedidoID)
        REFERENCES Pedidos(PedidoID)
        ON DELETE CASCADE
);