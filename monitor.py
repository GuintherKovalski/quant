# coding: utf-8
__author__ = "Juliano Barreto"

import mysql
from mysql.connector import errorcode
import asyncio
import websockets.client
import json
import requests
import time
import sys
import hmac
import hashlib
import base64
from urllib import parse
import datetime
import gzip
import urllib


# Classes:
class Monitora:
    def __init__(self, codigo_usuario):
        self.codigo_usuario = codigo_usuario
        self.algoritmo = []
        self.logado = False

    @staticmethod
    def conecta():
        conector = None
        try:
            conector = mysql.connector.connect(user='login', password='senha', host='localhost',
                                               database='database')
        except mysql.connector.Error as erro:
            if erro.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Opa, não consegui logar no banco de dados!")
            else:
                print(erro)
        return conector

    @staticmethod
    def grava_log(texto, arquivo="registro.txt"):
        localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(arquivo, "w") as text_file:
            print(f"{localtime} - {texto}\n", file=text_file)

    # Demais funções comuns.


class MonitoraHitbtc(Monitora):
    def __init__(self, codigo_usuario):
        super().__init__(codigo_usuario)
        self.uri = "wss://api.hitbtc.com/api/2/ws"

    @staticmethod
    def busca_moedas_hitbtc():
        url = 'https://api.hitbtc.com/api/2/public/symbol'
        r = requests.get(url)
        retorno = {}
        resposta = r.json()
        for moeda in resposta:
            retorno[moeda['id']] = {'simbolo': moeda['id'], 'moeda': moeda['baseCurrency'],
                                    'par': moeda['quoteCurrency'], 'incremento': float(moeda['quantityIncrement']),
                                    'tamanho': float(moeda['tickSize'])}
        return retorno

    @staticmethod
    def busca_dados_hitbtc(simbolo=''):
        url = "https://api.hitbtc.com/api/2/public/ticker/"+simbolo
        r = requests.get(url)
        retorno = {}
        resposta = r.json()

        if simbolo == '':
            for moeda in resposta:
                if moeda['ask'] is not None and moeda['bid'] is not None and moeda['volume'] is not None \
                        and moeda['volumeQuote'] is not None:
                    if moeda['open'] is None or moeda['open'] == '':
                        moeda['open'] = float(0.0)
                    if moeda['low'] is None or moeda['low'] == '':
                        moeda['low'] = 0
                    if moeda['last'] is None or moeda['last'] == '':
                        moeda['last'] = 0.0
                    if moeda['high'] is None or moeda['high'] == '':
                        moeda['high'] = 0.0

                    retorno[moeda['symbol']] = {'simbolo': moeda['symbol'], 'ask': float(moeda['ask']),
                                                'bid': float(moeda['bid']), 'last': float(moeda['last']),
                                                'open': float(moeda['open']), 'low': float(moeda['low']),
                                                'high': float(moeda['high']), 'volume': float(moeda['volume']),
                                                'volumeQuote': float(moeda['volumeQuote'])}
        else:
            if 'symbol' in resposta:
                retorno = {'simbolo': resposta['symbol'], 'ask': float(resposta['ask']),
                           'bid': float(resposta['bid']), 'volume': float(resposta['volume']),
                           'volumeQuote': float(resposta['volumeQuote'])}

        return retorno

    def monitora_hitbtc(self):
        async def hello():
            async with websockets.connect(self.uri) as websocket:
                while 1:
                    usuario = {'hitbtc_public_key': 'public key', 'hitbtc_private_key': 'pvt key', 'id': 'codigo'}
                    # Primeira requisição: Fazer login na exchange, caso não logado:
                    if self.logado is False:
                        j_login = {'method': 'login',
                                   'params': {'algo': 'BASIC', "pKey": usuario['hitbtc_public_key'],
                                              "sKey": usuario['hitbtc_private_key']}, 'id': usuario['id']}
                        await websocket.send(json.dumps(j_login))
                        data_login = await websocket.recv()
                        resposta_login = json.loads(data_login)
                        if resposta_login['result'] is True:
                            print("Logado no sistema! Conectando ao streaming...")
                            self.logado = True
                        else:
                            print("Erro ao logar no sistema! Cancelando...")
                            exit(0)

                    j_report = {"method": "subscribeReports", "params": {}}
                    requisicao = json.dumps(j_report)
                    await websocket.send(requisicao)
                    data_report = await websocket.recv()  # Retorna True
                    resposta_report = json.loads(data_report)
                    if resposta_report['result'] is True:
                        print("Conectado com sucesso!")
                        # Buscando as ordens existentes e liberando para escutar alterações.
                        # Na primeira conexão o sistema da HitBTC sempre retorna as últimas ordens:
                        data_ordens_hitbtc = await websocket.recv()
                        resposta_ordens_hitbtc = json.loads(data_ordens_hitbtc)
                        print(resposta_ordens_hitbtc)
                    else:
                        print("Erro ao se conectar! Abortando...")
                        exit(0)

                    while True:
                        print("\nOuvindo o servidor...")
                        data_ouvindo = await websocket.recv()
                        resposta_ouvindo = json.loads(data_ouvindo)
                        localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        print(f"\n ORDEM EM {localtime}: ")
                        print(f"-----------------------")
                        print(resposta_ouvindo)
                        if 'method' in resposta_ouvindo:
                            dados_ordem = resposta_ouvindo['params']
                        else:
                            continue  # Se não for ordem, ignora. De vez em quando a HitBTC envia trash data.

                        # Buscando informações da ordem (se é do bot ou foi imputada manualmente):
                        ordem = self.busca_dados_ordem({"codigo": dados_ordem['clientOrderId'], "exchange": "HitBTC"})
                        if len(ordem) <= 0:
                            print(f"Essa ordem ({dados_ordem['clientOrderId']}) não é minha. Ignorando...")
                            continue

                        # Atualizando as informações da ordem
                        # Status possíveis: new, suspended, partiallyFilled, filled, canceled, expired.
                        if dados_ordem['status'] == 'filled':

                            # Se a ordem for de compra:
                            if dados_ordem['side'] == 'buy':
                                # Aqui o algoritmo faz alguns testes, atualiza banco de dados e  emite ordem de venda.
                                continue

                            # Se a ordem for de venda:
                            elif dados_ordem['side'] == 'sell':
                                # Envia Pushbullet da venda finalizada e atualiza o banco de dados com as informações.
                                continue

                        elif dados_ordem['status'] == 'partiallyFilled':
                            # Tratamento da ordem parcial...
                            print("Esta ordem está parcialmente completa. Continuando...")

                        elif dados_ordem['status'] == 'canceled':
                            # Tratamento de ordem cancelada. Apenas ajuste no BD.
                            print("Esta ordem está cancelada. Continuando...")

                        elif dados_ordem['status'] == 'expired':
                            # Tratamento de ordem expirada. Nem cheguei a implementar porque minhas ordens não expiram.
                            print("Esta ordem expirou. Continuando...")

        asyncio.get_event_loop().run_until_complete(hello())


class MonitoraHuobi(Monitora):
    def __init__(self, codigo_usuario):
        super().__init__(codigo_usuario)
        self.uri = "wss://api.huobi.pro/ws/v1"

    @staticmethod
    def busca_moedas_huobi():
        url = 'https://api.huobi.pro/v1/common/symbols'
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        retorno = {}
        if r is not None:
            resposta = r.json()
            moedas = resposta['data']
            for moeda in moedas:
                decimais = (1 / 10 ** moeda['price-precision'])
                incremento = (1 / 10 ** moeda['amount-precision'])
                # print(format(decimais, f".{moeda['price-precision']}f"))
                retorno[moeda['symbol'].upper()] = {'simbolo': moeda['symbol'].upper(),
                                                    'moeda': moeda['base-currency'].upper(),
                                                    'par': moeda['quote-currency'].upper(),
                                                    'incremento': format(incremento, f".{moeda['amount-precision']}f"),
                                                    'tamanho': format(decimais, f".{moeda['price-precision']}f")}
        else:
            print(f"Erro ao buscar dados das moedas da Huobi. Erro: {r}")
        return retorno

    @staticmethod
    def busca_dados_huobi(simbolo=None):
        retorno = {}
        if simbolo is None:
            url = f"https://api.huobi.pro/v1/common/symbols"
            r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            if r is not None:
                resposta = r.json()
                simbolos = resposta['data']
                for s in simbolos:
                    url2 = f"https://api.huobi.pro/market/detail/merged?symbol={s['symbol']}"
                    r2 = requests.get(url2, headers={'User-Agent': 'Mozilla/5.0'})
                    resposta2 = r2.json()
                    if "tick" not in resposta2:
                        continue
                    moeda = resposta2['tick']
                    if ('ask' in moeda and moeda['ask'][0] is not None) and \
                            ('bid' in moeda and moeda['bid'][0] is not None) and \
                            ('vol' in moeda and moeda['vol'] is not None) and \
                            ('amount' in moeda and moeda['amount'] is not None):
                        if moeda['open'] is None or moeda['open'] == '':
                            moeda['open'] = float(0.0)
                        if moeda['low'] is None or moeda['low'] == '':
                            moeda['low'] = 0
                        if moeda['high'] is None or moeda['high'] == '':
                            moeda['high'] = 0.0

                        retorno[s['symbol'].upper()] = {'simbolo': s['symbol'].upper(),
                                                        'ask': float(moeda['ask'][0]),
                                                        'bid': float(moeda['bid'][0]),
                                                        'open': float(moeda['open']),
                                                        'low': float(moeda['low']),
                                                        'high': float(moeda['high']),
                                                        'volume': float(moeda['vol']),
                                                        'volumeQuote': float(moeda['amount'])}
            else:
                print(f"Erro ao buscar dados das moedas na Huobi. Erro: {r.status_code}")
        else:
            url = f"https://api.huobi.pro/market/detail/merged?symbol={simbolo.lower()}"
            r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            if r is not None:
                resposta = r.json()
                moeda = resposta['data']
                if ('ask' in moeda and moeda['ask'][0] is not None) and \
                        ('bid' in moeda and moeda['bid'][0] is not None) and \
                        ('vol' in moeda and moeda['vol'] is not None) and \
                        ('amount' in moeda and moeda['amount'] is not None):
                    retorno[simbolo.upper()] = {'simbolo': simbolo.upper(),
                                                'ask': moeda['ask'],
                                                'bid': moeda['bid'],
                                                'volume': moeda['vol'],
                                                'volumeQuote': moeda['amount']}
            else:
                print(f"Erro ao buscar dados da moeda {simbolo} na Huobi. Erro: {r.status_code}")

        return retorno

    @staticmethod
    def retorna_assinatura(chave_publica, chave_privada, timestamp):
        params = dict()
        params['SignatureMethod'] = "HmacSHA256"
        params['SignatureVersion'] = 2
        params['AccessKeyId'] = chave_publica
        params['Timestamp'] = urllib.parse.quote(timestamp)

        keys = sorted(params.keys())

        qs = '&'.join(['%s=%s' % (key, params[key]) for key in keys])
        print(qs)

        payload = '%s\n%s\n%s\n%s' % ('GET', "api.huobi.pro", "/v1", qs)
        print(payload)
        payload = urllib.parse.quote(payload)
        mensagem = payload.encode('utf-8')
        chave = chave_privada.encode('utf-8')
        dig = hmac.new(chave, mensagem, digestmod=hashlib.sha256).digest()

        # return base64.b64encode(dig).decode()
        return base64.b64encode(dig)

    def monitora_superminion_huobi(self):
        async def hello():
            async with websockets.connect(self.uri) as websocket:
                while 1:
                    usuario = {'hitbtc_public_key': 'public key', 'hitbtc_private_key': 'pvt key', 'id': 'codigo'}

                    # Primeira requisição: Fazer login na exchange, caso não logado:
                    if self.logado is False:
                        # huobi.web_socket.

                        timestamp2 = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
                        print(timestamp2)
                        j_ping = {"op": "ping"}
                        await websocket.send(json.dumps(j_ping))
                        data_pong = await websocket.recv()
                        pong = json.loads(gzip.decompress(data_pong).decode('utf-8'))
                        timestamp = datetime.datetime.utcfromtimestamp(pong['ts']/1000).strftime('%Y-%m-%dT%H:%M:%S')
                        print(timestamp)
                        assinatura = self.retorna_assinatura(usuario['huobi_public_key'],
                                                             usuario['huobi_private_key'],
                                                             timestamp)
                        print(assinatura)

                        j_login = {"op": "auth",
                                   "AccessKeyId": usuario['huobi_public_key'],
                                   "SignatureMethod": "HmacSHA256",
                                   "SignatureVersion": "2",
                                   "Timestamp": timestamp,
                                   "Signature": assinatura}

                        await websocket.send(json.dumps(j_login))
                        data_login = await websocket.recv()
                        # print(data_login)
                        dados = gzip.decompress(data_login).decode('utf-8')

                        print(dados)
                        exit()

                    j_report = {"method": "subscribeReports", "params": {}}
                    requisicao = json.dumps(j_report)
                    await websocket.send(requisicao)
                    data_report = await websocket.recv()  # Retorna True
                    print(data_report)
                    # resposta_report = json.loads(data_report)
                    # if resposta_report['result'] is True:
                    #     print("Conectado com sucesso!")
                    #     # Buscando as ordens existentes e liberando para escutar alterações:
                    #     data_ordens_huobi = await websocket.recv()
                    #     resposta_ordens_huobi = json.loads(data_ordens_huobi)
                    #     print(resposta_ordens_huobi)
                    # else:
                    #     print("Erro ao se conectar! Abortando...")
                    #     exit(0)

                    while True:
                        print("\nOuvindo o servidor...")
                        data_ouvindo = await websocket.recv()
                        resposta_ouvindo = json.loads(data_ouvindo)
                        localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        print(f"\n ORDEM EM {localtime}: ")
                        print(f"-----------------------")
                        print(resposta_ouvindo)
                        if 'method' in resposta_ouvindo:
                            dados_ordem = resposta_ouvindo['params']
                        else:
                            continue  # Se não for ordem, ignora.

                        # Atualizando as informações da ordem
                        # Status possíveis: new, suspended, partiallyFilled, filled, canceled, expired.
                        if dados_ordem['status'] == 'filled':

                            # Se a ordem for de compra:
                            if dados_ordem['side'] == 'buy':
                                continue

                            # Se a ordem for de venda:
                            elif dados_ordem['side'] == 'sell':
                                continue

                        elif dados_ordem['status'] == 'partiallyFilled':
                            print("Esta ordem está parcialmente completa. Continuando...")

                        elif dados_ordem['status'] == 'canceled':
                            print("Esta ordem está cancelada. Continuando...")

                        elif dados_ordem['status'] == 'expired':
                            print("Esta ordem expirou. Continuando...")

        asyncio.get_event_loop().run_until_complete(hello())


if len(sys.argv) <= 1:
    print("Erro! Você precisa passar o parâmetro código de usuário e o nome da Exchange a monitorar...")
    exit(0)
else:
    cod_usuario = sys.argv[1]
    # Cria a classe correta e executa monitoramento.
