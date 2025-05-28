import pandas as pd
import sympy as sp
import numpy as np
import math
from sympy.physics.continuum_mechanics.beam import Beam


# Lista de Materias, contendo as caracteristicas na seguinte ordem: [Nome do Aço, Limite de Escoamento Médio (MPA), Limite de Ruptura Médio(MPA),  Impacto Izod Médio (J), Massa Específica (kg/m3), Preço Médio (USD/kg)]

dados_materiais = {
    "Aço": [
        "SAE 1020 (Laminado a quente)", "SAE 1020 (Laminado a frio)", "SAE 1040 (Laminado a quente)",
        "SAE 1040 (Laminado a frio)", "SAE 1050 (Laminado a quente)", "SAE 1050 (Laminado a frio)",
        "AISI 4140", "AISI 4340", "AISI 1045 (Temperado e revenido)", "AISI 4145", "AISI 1144",
        "AISI 4150", "AISI 1060", "AISI 4130", "AISI 5140", "AISI 6150",
        "SAE 1045 (Laminado a quente)", "SAE 1045 (Laminado a frio)"
    ],
    "Método de Fabricação / Tratamento": [
        "Laminado a quente", "Laminado a frio", "Laminado a quente e normalizado", "Laminado a frio",
        "Laminado a quente", "Laminado a frio", "Temperado e revenido", "Temperado e revenido",
        "Temperado e revenido", "Temperado e revenido", "Laminado a frio", "Temperado e revenido",
        "Laminado a frio", "Temperado e revenido", "Temperado e revenido", "Temperado e revenido",
        "Laminado a quente", "Laminado a frio"
    ],
    "Limite de Escoamento (MPa)": [
        350, 380, 415, 450, 470, 500, 655, 710, 530, 760, 690, 825, 580, 440, 620, 700, 310, 340
    ],
    "Limite de Ruptura (MPa)": [
        440, 460, 580, 610, 630, 670, 1020, 1080, 760, 1040, 860, 1130, 750, 720, 870, 980, 565, 600
    ],
    "Limite de Ruptura Mínima (MPa)": [
        400, 420, 500, 520, 550, 570, 900, 950, 620, 950, 780, 1000, 680, 600, 750, 850, 540, 570
    ],
    "Limite de Escoamento Médio (MPa)": [
        345, 380, 415, 450, 470, 500, 655, 710, 530, 760, 690, 825, 580, 440, 620, 700, 310, 340
    ],
    "Limite de Ruptura Médio (MPa)": [
        440, 460, 580, 610, 630, 670, 1020, 1080, 760, 1040, 860, 1130, 750, 720, 870, 980, 565, 600
    ],
    "Dureza Brinell (HB)": [
        120, 130, 180, 190, 200, 210, 240, 260, 210, 260, 230, 290, 210, 217, 220, 250, 170, 180
    ],
    "Impacto Izod (J)": [
        34, 32, 25, 23, 22, 20, 35, 27, 28, 28, 30, 25, 18, 35, 22, 24, 26, 24
    ],
    "Impacto Izod Médio (J)": [
        34, 32, 25, 23, 22, 20, 35, 27, 28, 28, 30, 25, 18, 35, 22, 24, 26, 24
    ],
    "Massa Específica (kg/m³)": [
        7850, 7850, 7850, 7850, 7850, 7850, 7850, 7850, 7850, 7850, 7850, 7850, 7850, 7850, 7850, 7850, 7850, 7850
    ],
    "Preço Médio (USD/kg)": [
        0.80, 0.90, 1.00, 1.00, 1.20, 1.30, 1.50, 1.80, 1.30, 1.60, 1.30, 1.70, 1.30, 1.40, 1.50, 1.60, 1.00, 1.10
    ]
}

baseMateriais = pd.DataFrame(dados_materiais)
baseMateriais = baseMateriais[['Aço',
                              'Método de Fabricação / Tratamento',
                              'Limite de Escoamento Médio (MPa)',
                              'Limite de Ruptura Médio (MPa)',
                              'Impacto Izod Médio (J)',
                              'Massa Específica (kg/m³)',
                              'Preço Médio (USD/kg)']]
dicMateriais = baseMateriais.to_dict('records')


# Lista base chavetas

dados_chavetas = {
    "Diâmetro Eixo Min (mm)": [6, 8, 10, 12, 17, 22, 30, 38, 44, 50, 58, 65, 75, 85, 95],
    "Diâmetro Eixo Max (mm)": [8, 10, 12, 17, 22, 30, 38, 44, 50, 58, 65, 75, 85, 95, 100],
    "Largura b (mm)":         [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 28],
    "Altura h (mm)":          [2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 14, 14, 16],
    "Profundidade no eixo t1 (mm)":   [1.20, 1.80, 2.50, 3.00, 3.50, 4.00, 5.00, 5.00, 5.50, 6.00, 7.00, 7.50, 8.00, 8.50, 9.00],
    "Profundidade no cubo t2 (mm)":   [1.00, 1.40, 1.80, 2.30, 2.80, 3.30, 3.30, 3.30, 3.80, 4.30, 4.40, 4.60, 5.00, 5.50, 6.00],
    "Comprimento l (mm)": [
        "10 - 50", "12 - 50", "14 - 60", "18 - 70", "20 - 80",
        "25 - 90", "28 - 100", "32 - 110", "36 - 125", "40 - 140",
        "45 - 150", "50 - 160", "56 - 180", "63 - 200", "70 - 220"
    ]
}

tabChavetas = pd.DataFrame(dados_chavetas)



#Definindo Funções

#Funções para calcular dimensões iniciais

#Função para calcular o torque que o eixo será submetido
def TorqueEixo(Potencia, rotacao, K):
  #Potencia em Watts
  #K = fator de sobrecarga (partida, operação, etc)

  torque = (Potencia*60*K)/(2*math.pi*rotacao)
  return torque


#Função para calcular as forças oriundas da transmissão por engrenagens cilindricas de dentes retos:
def CargaEngrenagemDentesRetos(torque, modulo, numdentes, angpressao):
  #torque em N.m
  #modulo = modulo da engrenagem
  #numdentes = numero de dentes da engrenagem
  #angpessão = angulo de pressão da engrenagem (em graus)

  angpressao = math.radians(angpressao)
  ForçaTangencial = (2*torque*1000)/(modulo*numdentes)
  ForçaNormal = ForçaTangencial*math.tan(angpressao)
  listaForçasEngrenagem = [0, ForçaTangencial, ForçaNormal]
  return listaForçasEngrenagem


#Função para calcular as forças oriundas da transmissão por polia/correia:
def CargaPoliaCorreia(torque, f, theta, Fc, Diametro):
  #torque em N.m
  #f = coeficiente de atrito
  #theta = angulo de abraçamento das correias em graus
  #Diametro = diametro da polia (em milimetros)

  F1, F2, T, F, Theta, Fcent, D = sp.symbols('F1 F2 T F theta Fc D')

  T = torque
  F = f
  Theta = math.radians(theta)
  Fcent = Fc
  D = Diametro
  listaForçasPoliaCorreia = list(list(sp.solve(((F1 - Fc) - (math.exp(f*Theta)*(F2-Fcent)), F1 - F2 - ((2*T*1000)/D)), [F1, F2], force=True, manual=True, set=True)[1])[0])
  listaForçasPoliaCorreia = [0] + listaForçasPoliaCorreia
  return listaForçasPoliaCorreia


def GeracaoInicial(ForcaEntrada, ForcaSaida, dicMateriais, FSinicial):
  #Função para determinar a geração inicial para inicio dos calculos

  #ForcaEntrada = lista com as forças na seção 1 [F1x, F1y, F1z]
  #ForcaSaida = lista com as forças na seção 4 [F4x, F4y, F4z]
  #dicMateriais = lista de dicionários com os materiais possíveis e suas propriedades
  #FSinicial = fator de segurança estático inicial

  forcazyEntrada = math.sqrt((ForcaEntrada[1])**2 + (ForcaEntrada[2])**2) # módulo da resultante no plano yz
  forcazySaida = math.sqrt((ForcaSaida[1])**2 + (ForcaSaida[2])**2) #módulo da resultante no plano yz

  forcaReferenciaEntrada = max([abs(ForcaEntrada[0]), forcazyEntrada])
  forcaReferenciaSaida = max([abs(ForcaSaida[0]), forcazySaida])

  df_GeracaoInicial = pd.DataFrame(columns=['Material',
                                          'E (GPa)',
                                          'Método de Fabricação / Tratamento',
                                          'Limite de Escoamento (MPa)',
                                          'Limite de Ruptura (MPa)',
                                          'Diametro S1',
                                          'Diametro S2',
                                          'Diametro S3',
                                          'Diametro S4',
                                          'Diametro S5',
                                          'Diametro S6',
                                          'Comprimento S1',
                                          'Comprimento S2',
                                          'Comprimento S3',
                                          'Comprimento S4',
                                          'Comprimento S5',
                                          'Comprimento S6',
                                          'Massa Específica (kg/m³)',
                                          'Preço Médio (USD/kg)'])

  for i in range(100):

    material = np.random.choice(dicMateriais)
    listaFatorMultiplicador = [5,6,7,8,9,10,11,12,13,14,15]
    fatorMultiplicador = np.random.choice(listaFatorMultiplicador)

    Diam1 = round(math.sqrt((4*FSinicial*forcaReferenciaEntrada)/(math.pi*material['Limite de Escoamento Médio (MPa)']))*fatorMultiplicador,2)

    Diam4 = round(math.sqrt((4*FSinicial*forcaReferenciaSaida)/(math.pi*material['Limite de Escoamento Médio (MPa)']))*fatorMultiplicador,2)

    distanciaIncremento = (Diam4-Diam1)/3

    Diam2 = round(((Diam1)/2 + distanciaIncremento)*2,2)
    Diam3 = round(((Diam1)/2 + 1.2*distanciaIncremento)*2,2)
    Diam5 = 1.1*Diam4
    Diam6 = Diam3

    # Caso queira comprimentos randomicos com base no diâmetro da seção
    '''L1 = round(np.random.choice(np.linspace(Diam1, 2*Diam1, num=20 ,endpoint=False)[1:]),2)
    L2 = round(np.random.choice(np.linspace(Diam2, 2*Diam2, num=20 ,endpoint=False)[1:]),2)
    L3 = round(np.random.choice(np.linspace(Diam3, 2*Diam3, num=20 ,endpoint=False)[1:]),2)
    L4 = round(np.random.choice(np.linspace(Diam4, 2*Diam4, num=20 ,endpoint=False)[1:]),2)
    L5 = round(np.random.choice(np.linspace(Diam5, 2*Diam5, num=20 ,endpoint=False)[1:]),2)'''

# Neste projeto, trabalhamos com comprimentos de seção fixos
    L1 = 31
    L2 = 18
    L3 = 18
    L4 = 35
    L5 = 5
    L6 = 18

    E = 210    #Definindo modulo de elasticidade = 210 GPa para todos os aços

    L = [[material['Aço'],
          E,
          material['Método de Fabricação / Tratamento'],
          material['Limite de Escoamento Médio (MPa)'],
          material['Limite de Ruptura Médio (MPa)'],
          Diam1,
          Diam2,
          Diam3,
          Diam4,
          Diam5,
          Diam6,
          L1,
          L2,
          L3,
          L4,
          L5,
          L6,
          material['Massa Específica (kg/m³)'],
          material['Preço Médio (USD/kg)']]]
    df = pd.DataFrame(L, columns=['Material',
                                'E (GPa)',
                                'Método de Fabricação / Tratamento',
                                'Limite de Escoamento (MPa)',
                                'Limite de Ruptura (MPa)',
                                'Diametro S1',
                                'Diametro S2',
                                'Diametro S3',
                                'Diametro S4',
                                'Diametro S5',
                                'Diametro S6',
                                'Comprimento S1',
                                'Comprimento S2',
                                'Comprimento S3',
                                'Comprimento S4',
                                'Comprimento S5',
                                'Comprimento S6',
                                'Massa Específica (kg/m³)',
                                'Preço Médio (USD/kg)'])

    df_GeracaoInicial = pd.concat([df_GeracaoInicial, df], axis=0)

  return df_GeracaoInicial


def momentoAlternado(Mmax, Mmin, Diametro):
  Malternado = (32*((Mmax - Mmin)/2))/((Diametro**3)*math.pi)
  return Malternado


def torqueAlternado(Tmax, Tmin, Diametro):
  Talternado = (16*((Tmax - Tmin)/2))/((Diametro**3)*math.pi)
  return Talternado


def momentoMedio(Mmax, Mmin, Diametro):
  Mmedio = (32*((Mmax + Mmin)/2))/((Diametro**3)*math.pi)
  return Mmedio


def torqueMedio(Tmax, Tmin, Diametro):
  Tmedio = (16*((Tmax + Tmin)/2))/((Diametro**3)*math.pi)
  return Tmedio


def calcular_kb(diametro):
    """
    Calcula o fator de tamanho (kb) para um eixo circular com base no diâmetro.

    Parâmetros:
    - diametro (float): Diâmetro do eixo em milímetros.

    Retorno:
    - kb (float): Fator de tamanho.
    """
    if 2.79 <= diametro <= 51:
        # Intervalo 1: 2,79 mm <= d <= 51 mm
        kb = ((diametro/7.62) ** -0.107)
    elif 51 < diametro <= 254:
        # Intervalo 2: 51 mm < d <= 254 mm
        kb = 1.51 * (diametro ** -0.157)
    else:
        raise ValueError("Diâmetro fora do intervalo permitido (2.79 mm <= d <= 254 mm).")

    return kb


def tensaoFadiga(ka, kb, kc, kd, ke, kf, tensaoRuptura):
  TensaoFadiga = ka*kb*kc*kd*ke*kf*(tensaoRuptura/2)
  return TensaoFadiga


def Kt(D, d, r):
    h = (D - d) / 2
    a = h / r

    # Momento fletor
    if 0.25 <= a <= 2.0:
        C1 = 0.927 + 1.149 * (a ** 0.5) - 0.086 * a
        C2 = 0.015 - 3.281 * (a ** 0.5) + 0.837 * a
        C3 = 0.847 + 1.716 * (a ** 0.5) - 0.506 * a
        C4 = -0.790 + 0.417 * (a ** 0.5) - 0.246 * a
        ktfletor = C1 + C2 * (2 * h / D) + C3 * ((2 * h / D) ** 0.5) + C4 * ((2 * h / D) ** (1 / 3))
    elif 2.0 < a <= 20:
        C1 = 1.225 + 0.831 * (a ** 0.5) - 0.010 * a
        C2 = -3.790 + 0.958 * (a ** 0.5) - 0.257 * a
        C3 = 7.374 - 4.834 * (a ** 0.5) + 0.862 * a
        C4 = -3.809 + 3.046 * (a ** 0.5) - 0.595 * a
        ktfletor = C1 + C2 * (2 * h / D) + C3 * ((2 * h / D) ** 0.5) + C4 * ((2 * h / D) ** (1 / 3))
    else:
        ktfletor = np.nan  # Valor inválido

    # Momento torsor
    if 0.25 <= a <= 4.0:
        C1 = 0.953 + 0.680 * (a ** 0.5) - 0.053 * a
        C2 = -0.493 - 1.820 * (a ** 0.5) + 0.517 * a
        C3 = 1.621 + 0.908 * (a ** 0.5) - 0.529 * a
        C4 = -1.081 + 0.232 * (a ** 0.5) + 0.065 * a
        kttorsor = C1 + C2 * (2 * h / D) + C3 * ((2 * h / D) ** 0.5) + C4 * ((2 * h / D) ** (1 / 3))
    else:
        kttorsor = np.nan  # Valor inválido

    return [ktfletor, kttorsor]


def constanteNeuber(Sut):
  # constante utilizada para encontrar o q (fator de sensibilidade ao entalhe, pela curva de ajuste)
  # Sut deve ser dado em MPa
  Sut = Sut*145.038       #conversão para psi
  contanteNeuberFlexao = 0.246 - 3.08*(10**(-3))*Sut + 1.51*(10**(-5))*(Sut**2) - 2.67*(10**(-8))*(Sut**3)
  contanteNeuberTorçao = 0.190 - 2.15*(10**(-3))*Sut + 1.35*(10**(-5))*(Sut**2) - 2.67*(10**(-8))*(Sut**3)
  return [contanteNeuberFlexao, contanteNeuberTorçao]


def q(constNeuber, raioEntalhe):
  q = 1/(1 + (constNeuber/raioEntalhe)**(0.5))
  return q


def Kff(q, Ktf):
  kff = 1 + q*(Ktf-1)
  return kff


def Kft(q, Ktt):
  kft = 1 + q*(Ktt-1)
  return kft


def tensaoAlternada(Kff, Kft, tensaoNormalAlternada, tensaoCisalhanteAlternada):
  sigmaA = ((Kff*tensaoNormalAlternada)**2 + 3*(Kft*tensaoCisalhanteAlternada))**0.5
  return sigmaA


def tensaoMedia(tensaoNormalMedia, tensaoCisalhanteMedia):
  sigmaA = ((tensaoNormalMedia)**2 + 3*(tensaoCisalhanteMedia))**0.5
  return sigmaA


def CritGoldman(Talternada, Tmedia, Tfadiga, Truptura, FSfadiga):
  import sympy as sp
  Talter, Tmed, Tfad, Trup, FSfad = sp.symbols('Taltr, Tmed, Tfad, Trup, FSfad')
  Talter = Talternada
  Tmed = Tmedia
  Tfad = Tfadiga
  Trup = Truptura
  exprGoldman = (Talter/Tfad) + (Tmed/Trup) - (1/FSfad)
  solveGoldman = sp.solve(exprGoldman, FSfad)
  if solveGoldman[0] > FSfadiga:
    dimensionamento = "OK"
  else:
    dimensionamento = f"Mal dimensionado para fadiga"
  return [dimensionamento, solveGoldman[0]]


def CalcChaveta(Torque, Diametro, bCubo, hCubo, t1, t2, CompRasgo):
  #Torque em N*mm
  #Diametro = diametro do eixo em mm
  #hCubo = altura h da chavetaa
  #t1 e t2 são as dimensões laterais da chaveta, em relação a interface eixo-cubo, em mm
  #CompRasgo = é o comprimento previsto da chaveta, na direção longitudinal ao eixo, em mm

  if any(pd.isnull([Torque, Diametro, bCubo, hCubo, t1, CompRasgo])):
    return [None, "Chavetas mal dimensionadas."]

  fatorRestriçãoAçoCuboAço = 0.6*150    # Valor Básico de pressão p0 de acordo com o material dos cubos [MPa], choques fortes, um flanco
  CompEfetivo = CompRasgo - bCubo
  p1 = (2*Torque)/((hCubo - t1)*CompEfetivo*Diametro)
  tipo = "1 Chaveta"
  if p1 > fatorRestriçãoAçoCuboAço:
    p1 = (2*Torque)/((hCubo - t1)*CompEfetivo*2*0.75*Diametro)
    tipo = "2 Chavetas"
  if p1 > fatorRestriçãoAçoCuboAço:
    resultado = [None,"Chavetas mal dimensionadas."]
  else:
    resultado = [p1, tipo]
  return resultado


# Função para fazer o merge das dimensões das chavetas
def get_chaveta_dimensions(diametro, tab_chavetas):
    # Filtra a tabela de chavetas para encontrar o menor valor de Diâmetro Eixo Min que seja maior que o diâmetro
    row = tab_chavetas[tab_chavetas['Diâmetro Eixo Min (mm)'] < diametro].sort_values(by='Diâmetro Eixo Min (mm)',ascending=False).head(1)

    # Se encontrar uma linha, usa o intervalo imediatamente menor, ou seja, a linha anterior
    if not row.empty:
        index = row.index[0]
        if index > 0:
            # Pega a linha imediatamente anterior
            result = tab_chavetas.iloc[index - 1]
        else:
            # Caso não exista linha anterior (diâmetro muito pequeno), retorna a primeira linha
            result = row.iloc[0]
    else:
        result = None

    # Retorna um dicionário com valores, preenchendo com None caso não haja resultado
    return {
        'Largura b (mm)': result['Largura b (mm)'] if result is not None else None,
        'Altura h (mm)': result['Altura h (mm)'] if result is not None else None,
        'Profundidade no eixo t1 (mm)': result['Profundidade no eixo t1 (mm)'] if result is not None else None,
        'Profundidade no cubo t2 (mm)': result['Profundidade no cubo t2 (mm)'] if result is not None else None,
        'Comprimento l (mm)': result['Comprimento l (mm)'] if result is not None else None
    }


def calcular_Ieq(df):
    """
    Calcula o momento de inércia equivalente (Ieq) para cada linha do DataFrame e adiciona como nova coluna.

    Parâmetros:
    - df: DataFrame contendo colunas 'Diametro S1' a 'Diametro S6',
          'Comprimento S1' a 'Comprimento S6', e 'E (GPa)'.

    Retorno:
    - DataFrame original com nova coluna 'Ieq (m⁴)'.
    """

    mm4_to_m4 = 1e-12
    Ieq_list = []

    for _, row in df.iterrows():
        Ieq_inv_sum = 0
        for i in range(1, 7):
            d = row[f'Diametro S{i}']
            L = row[f'Comprimento S{i}']
            E = row['E (GPa)'] * 1e9  # GPa -> Pa
            I = (np.pi / 64) * (d ** 4) * mm4_to_m4  # mm⁴ -> m⁴
            Ieq_inv_sum += (L / 1000) / (E * I)  # L em m

        Ieq = 1 / Ieq_inv_sum if Ieq_inv_sum != 0 else np.nan
        Ieq_list.append(Ieq)

    df = df.copy()
    df['Ieq (m^4)'] = Ieq_list
    return df



# CÓDIGOS DA PARTE DE OTIMIZAÇÃO

# AVALIAÇÕES RESTRITIVAS (restrições)

def criterio_von_mises(FS_estatico, M, T, d, limite_escoamento):
    """
    Verifica se o ponto avaliado atende ao critério de von Mises para falha estática.

    Parâmetros:
    - FS_estatico: Fator de segurança mínimo exigido.
    - M: Momento fletor na seção [Nmm]
    - T: Torque aplicado na seção [Nmm]
    - d: Diâmetro da seção [mm]
    - limite_escoamento: Tensão de escoamento do material [MPa]

    Retorna:
    - True se atender ao critério de von Mises, False caso contrário.
    """
    # Tensão normal (MPa)
    sigma = (32 * M) / (sp.pi * d**3)  # [Nmm/mm^3 = MPa]
    # Tensão cisalhante (MPa)
    tau = (16 * T) / (sp.pi * d**3)    # [Nmm/mm^3 = MPa]

    # Tensão equivalente de von Mises
    sigma_vm = sp.sqrt(sigma**2 + 3 * tau**2)

    # Fator de segurança real
    FS_real = limite_escoamento / sigma_vm

    return FS_real >= FS_estatico, float(FS_real), float(sigma_vm)


def avaliacao_restritiva(df_input, MomentosS3, MomentosS4, TorqueMaxPartida, TorqueOperacao, FSfadiga, FSestatico, r):
    """
    Função para avaliar se os indivíduos no DataFrame atendem às restrições do critério de Goodman,
    do critério de Von Mises e ao critério de chavetas.
    
    Parâmetros:
    - df_input: DataFrame com as soluções.
    - MomentosS3: [Mmax, Mmin] da seção S3
    - MomentosS4: [Mmax, Mmin] da seção S4
    - TorqueMaxPartida: Torque máximo aplicado
    - TorqueOperacao: Torque de operação
    - FSfadiga: Fator de segurança mínimo para fadiga
    - FSestatico: Fator de segurança mínimo para falha estática (von Mises)
    - r: Raio do entalhe do eixo

    Retorno:
    - DataFrame atualizado com resultado das restrições
    """
    df_input = pd.concat(
        [df_input,
         df_input['Diametro S1'].apply(lambda d: pd.Series(get_chaveta_dimensions(d, tabChavetas))).add_suffix(' S1')],
        axis=1
    )
    df_input = pd.concat(
        [df_input,
         df_input['Diametro S4'].apply(lambda d: pd.Series(get_chaveta_dimensions(d, tabChavetas))).add_suffix(' S4')],
        axis=1
    )

    df_input['Restrição Atendida'] = ''

    for index, row in df_input.iterrows():
        DiametroS1 = row['Diametro S1']
        DiametroS3 = row['Diametro S3']
        DiametroS4 = row['Diametro S4']
        LimiteEscoamento = row['Limite de Escoamento (MPa)']
        LimiteRuptura = row['Limite de Ruptura (MPa)']

        # Chaveta
        t1S1 = row['Profundidade no eixo t1 (mm) S1']
        t1S4 = row['Profundidade no eixo t1 (mm) S4']
        t2S1 = row['Profundidade no cubo t2 (mm) S1']
        t2S4 = row['Profundidade no cubo t2 (mm) S4']
        b1 = row['Largura b (mm) S1']
        b4 = row['Largura b (mm) S4']
        h1 = row['Altura h (mm) S1']
        h4 = row['Altura h (mm) S4']
        compS1 = 23
        compS4 = 30

        # Kt para entalhes em S3
        resultado_Kt = Kt(DiametroS4, DiametroS3, r)
        if isinstance(resultado_Kt[0], str):
            df_input.at[index, 'Restrição Atendida'] = 'Não Atende'
            continue
        Ktf, Ktt = resultado_Kt

        constNeuberFlexao, constNeuberTorcao = constanteNeuber(LimiteRuptura)
        qf = q(constNeuberFlexao, r)
        qt = q(constNeuberTorcao, r)

        Kff_S3 = Kff(qf, Ktf)
        Kft_S3 = Kft(qt, Ktt)
        Kff_S4 = 1.6
        Kft_S4 = 1.3

        MmaxS3, MminS3 = MomentosS3
        MmaxS4, MminS4 = MomentosS4

        ka_S3 = 1.58 * (LimiteRuptura ** -0.085)
        kb_S3 = calcular_kb(DiametroS3)
        ka_S4 = 4.51 * (LimiteRuptura ** -0.265)
        kb_S4 = calcular_kb(DiametroS4)

        kc = 0.814
        kd = 1
        ke = 1
        kf = 1

        TensaoFadigaS3 = tensaoFadiga(ka_S3, kb_S3, kc, kd, ke, kf, LimiteRuptura)
        TensaoFadigaS4 = tensaoFadiga(ka_S4, kb_S4, kc, kd, ke, kf, LimiteRuptura)

        MmedioS3 = momentoMedio(MmaxS3, MminS3, DiametroS3)
        MmedioS4 = momentoMedio(MmaxS4, MminS4, DiametroS4)
        TmedS3 = torqueMedio(TorqueOperacao, 0, DiametroS3)
        TmedS4 = torqueMedio(TorqueOperacao, 0, DiametroS4)

        MalternadoS3 = momentoAlternado(MmaxS3, MminS3, DiametroS3)
        MalternadoS4 = momentoAlternado(MmaxS4, MminS4, DiametroS4)
        TaltS3 = torqueAlternado(TorqueOperacao, 0, DiametroS3)
        TaltS4 = torqueAlternado(TorqueOperacao, 0, DiametroS4)

        # von Mises (estático) na pior condição: torque de partida
        atende_mises_S3, FS_real_S3, _ = criterio_von_mises(MmaxS3, TorqueMaxPartida, DiametroS3, LimiteEscoamento, FSestatico)
        atende_mises_S4, FS_real_S4, _ = criterio_von_mises(MmaxS4, TorqueMaxPartida, DiametroS4, LimiteEscoamento, FSestatico)

        # Goodman
        resultadoGoodmanS3 = CritGoldman(TaltS3, TmedS3, TensaoFadigaS3, LimiteRuptura, FSfadiga)
        resultadoGoodmanS4 = CritGoldman(TaltS4, TmedS4, TensaoFadigaS4, LimiteRuptura, FSfadiga)

        # Chavetas
        resultadoChavetaS1 = CalcChaveta(TorqueMaxPartida, DiametroS1, b1, h1, t1S1, t2S1, compS1)
        resultadoChavetaS4 = CalcChaveta(TorqueMaxPartida, DiametroS4, b4, h4, t1S4, t2S4, compS4)

        if (resultadoGoodmanS3[0] == "OK" and resultadoGoodmanS4[0] == "OK" and
            resultadoChavetaS1 != "Chavetas mal dimensionadas." and
            resultadoChavetaS4 != "Chavetas mal dimensionadas." and
            atende_mises_S3 and atende_mises_S4):
            df_input.at[index, 'Restrição Atendida'] = 'Atende'
        else:
            df_input.at[index, 'Restrição Atendida'] = 'Não Atende'

    return df_input


# Função para checar restrição de deflexão máxima e ângulo máximo
def checar_restricao_deflexao(Ematerial, Ieixo, comprimentoEixo, pontoMancal1, pontoMancal2,
                               listaForcasXY, listaPontoForcasXY, listaForcasXZ, listaPontoForcasXZ):
    """
    Verifica se a deflexão e ângulo estão dentro dos limites admissíveis:
    - Deflexão máxima ≤ 0.05 mm
    - Ângulo máximo ≤ 4 minutos
    """
    x = sp.symbols('x')
    E = Ematerial
    I = Ieixo

    def configurar_viga(forcas, posicoes, R1, R2):
        b = Beam(comprimentoEixo, E, I)
        b.apply_load(R1, pontoMancal1, -1)
        b.apply_load(R2, pontoMancal2, -1)
        for f, p in zip(forcas, posicoes):
            b.apply_load(f, p, -1)
        b.bc_deflection = [(pontoMancal1, 0), (pontoMancal2, 0)]
        b.solve_for_reaction_loads(R1, R2)
        return b

    # Criar e resolver vigas
    R1xy, R2xy = sp.symbols('R1xy R2xy')
    b1 = configurar_viga(listaForcasXY, listaPontoForcasXY, R1xy, R2xy)
    defl_xy = b1.deflection()
    slope_xy = b1.slope()

    R1xz, R2xz = sp.symbols('R1xz R2xz')
    b2 = configurar_viga(listaForcasXZ, listaPontoForcasXZ, R1xz, R2xz)
    defl_xz = b2.deflection()
    slope_xz = b2.slope()

    # Deflexão máxima (m → mm)
    deflexao_max_xy = float(max([abs(defl_xy.subs(x, xi).evalf()) for xi in np.linspace(0, comprimentoEixo, 100)])) * 1000
    deflexao_max_xz = float(max([abs(defl_xz.subs(x, xi).evalf()) for xi in np.linspace(0, comprimentoEixo, 100)])) * 1000

    # Ângulo máximo (rad → minutos)
    angulo_max_xy = float(max([abs(slope_xy.subs(x, xi).evalf()) for xi in np.linspace(0, comprimentoEixo, 100)])) * (180 * 60 / np.pi)
    angulo_max_xz = float(max([abs(slope_xz.subs(x, xi).evalf()) for xi in np.linspace(0, comprimentoEixo, 100)])) * (180 * 60 / np.pi)

    deflexao_total = max(deflexao_max_xy, deflexao_max_xz)
    angulo_total = max(angulo_max_xy, angulo_max_xz)

    if deflexao_total > 0.05 or angulo_total > 4:  #FLEXA MÁXIMA DE 0,5mm e angulo máximo de 5'
        return "Não Atende"
    else:
        return "Atende"


# Filtro de deflexão
def filtrar_por_deflexao(df):
    """
    Aplica a verificação de deflexão e ângulo máximo a cada linha do DataFrame,
    e adiciona uma coluna 'Restrição Deflexão' com 'Atende' ou 'Não Atende'.
    Retorna apenas as linhas que atendem à restrição.
    """
    df = calcular_Ieq(df)

    def verificar(row):
        Ematerial = row['E (GPa)'] * 1e9
        Ieixo = row['Ieq (m^4)']
        comprimentoEixo = sum([row[f'Comprimento S{i}'] for i in range(1, 7)]) / 1000
        pontoMancal1 = 0.058  #local fixo posição mancal 1 (S3)
        pontoMancal2 = 0.116  #local fixo posição mancal 1 (S6)

        centroS1 = row['Comprimento S1'] / 2 / 1000
        centroS4 = (sum([row[f'Comprimento S{i}'] for i in range(1, 4)]) + row['Comprimento S4'] / 2) / 1000

        return checar_restricao_deflexao(
            Ematerial,
            Ieixo,
            comprimentoEixo,
            pontoMancal1,
            pontoMancal2,
            listaForcasXY=[fEntrada[1],fSaida[1]],
            listaPontoForcasXY=[centroS1, centroS1],
            listaForcasXZ=[fEntrada[2],fSaida[2]],
            listaPontoForcasXZ=[centroS4, centroS4]
        )

    # Aplica a função linha a linha de forma vetorizada
    df = df.copy()
    df['Restrição Deflexão'] = df.apply(verificar, axis=1)

    # Filtra apenas as que atendem
    return df[df['Restrição Deflexão'] == 'Atende'].drop(columns=['Restrição Deflexão'])


# Função auxiliar para filtrar indivíduos com razões "a" fora do escopo de Kt
def filtrar_por_a_valido(df, r):
    """
    Filtra o DataFrame mantendo apenas os indivíduos cujas razões a = h/r
    (para transições S1→S2 e S4→S5) estejam no intervalo 0.25 <= a <= 4.0

    Parâmetros:
    - df: DataFrame contendo colunas 'Diametro S1', 'Diametro S2', 'Diametro S4', 'Diametro S5'
    - r: raio do entalhe

    Retorna:
    - DataFrame filtrado
    """
    df_filtrado = df.copy()

    # Calculando h e a para as seções S1->S2 e S4->S5
    df_filtrado["hS1S2"] = (df_filtrado["Diametro S2"] - df_filtrado["Diametro S1"]) / 2
    df_filtrado["hS4S5"] = (df_filtrado["Diametro S5"] - df_filtrado["Diametro S4"]) / 2

    df_filtrado["aS1S2"] = df_filtrado["hS1S2"] / r
    df_filtrado["aS4S5"] = df_filtrado["hS4S5"] / r

    # Aplicando o filtro
    cond_S1S2 = (df_filtrado["aS1S2"] > 0.25) & (df_filtrado["aS1S2"] < 4.0)
    cond_S4S5 = (df_filtrado["aS4S5"] > 0.25) & (df_filtrado["aS4S5"] < 4.0)

    return df_filtrado[cond_S1S2 & cond_S4S5].drop(columns=["hS1S2", "hS4S5", "aS1S2", "aS4S5"])


def filtrar_por_razao_D_L(df):
    """
    Filtra o DataFrame mantendo apenas os indivíduos cuja razão L/D esteja entre 1 e 2
    para as seções com ligação eixo-cubo (S1 e S4).
    """
    secoes_validas = []
    secoes = ['S1','S4']

    for s in secoes:
        diametro = df[f'Diametro {s}']
        comprimento = df[f'Comprimento {s}']
        razao = comprimento / diametro
        secoes_validas.append((razao >= 1) & (razao <= 2))

    # Combina todas as condições com AND lógico
    condicao_geral = secoes_validas[0]
    for cond in secoes_validas[1:]:
        condicao_geral &= cond

    return df[condicao_geral].copy()


# AVALIAÇÃO QUANTITATIVA (fitness)

def compute_mass_cost_ratio(row):
    # Obtendo densidade (kg/m³) e preço (USD/kg)
    densidade = row['Massa Específica (kg/m³)']
    preco = row['Preço Médio (USD/kg)']

    # Extraindo diâmetros e comprimentos em mm
    diametros_mm = [row['Diametro S1'], row['Diametro S2'], row['Diametro S3'], row['Diametro S4'], row['Diametro S5'], row['Diametro S6']]
    comprimentos_mm = [row['Comprimento S1'], row['Comprimento S2'], row['Comprimento S3'], row['Comprimento S4'], row['Comprimento S5'], row['Comprimento S6']]

    massa_total = 0.0
    custo_total = 0.0

    # Calculando a massa e o custo por seção
    for d_mm, L_mm in zip(diametros_mm, comprimentos_mm):
        # Converter mm para m
        d_m = d_mm * 0.001
        L_m = L_mm * 0.001

        # Volume do cilindro: V = π * (r²) * L
        r_m = d_m / 2.0
        area = np.pi * (r_m**2)
        volume = area * L_m  # em m³

        # Massa da seção em kg
        massa_secao = volume * densidade
        # Custo da seção em USD
        custo_secao = massa_secao * preco

        massa_total += massa_secao
        custo_total += custo_secao

    # Calcula massa*custo
    if custo_total != 0:
        massa_custo = massa_total * custo_total
    else:
        massa_custo = np.inf

    return pd.Series({
        'Massa Total': massa_total,
        'Custo Total': custo_total,
        'Massa*Custo': massa_custo
    })

def fitness(df):
    # Aplica a função a cada linha do DataFrame, sem iterrows
    df = df.copy()
    resultados = df.apply(compute_mass_cost_ratio, axis=1)
    df = pd.concat([df, resultados], axis=1)
    df = df.sort_values(by='Massa*Custo', ascending=True)
    return df



# CONDIÇÕES INICIAIS

PotenciaMotor = 3000  #Watts
rpm = 1280
Ko = 1.5
Kp = 2.5

#Engrenagem
móduloEng = 3
numDentes = 20
angPressão = 20

#Polia / Correia
coefAtrito = 0.4
angAbraçamento = 180
diamPolia = 160
fc = 0


#geração inicial
T = TorqueEixo(PotenciaMotor, rpm, Kp)
fSaida = CargaEngrenagemDentesRetos(T, móduloEng, numDentes, angPressão)
fEntrada = CargaPoliaCorreia(T, coefAtrito, angAbraçamento, fc, diamPolia)

df_GeracaoInicial = GeracaoInicial(fEntrada, fSaida, dicMateriais, 3)


#lista com os momentos e torques máximos e mínimos nas seçoes do mancal e engrenagem (em N.mm)
listaMomentosS3 = [54180, -54180]
listaMomentosS4 = [(15400**2 + 21480**2)**0.5, -(15400**2 + 21480**2)**0.5]
torqueOperaçao = 33600
torquePartidaMax = 56000


# ALGORITMO NSGA II

def nsga2_otimizacao(df_geracao_inicial,
                     ForcaEntrada,
                     ForcaSaida,
                     dicMateriais,
                     MomentosS3,
                     MomentosS4,
                     TorqueMaxPartida,
                     TorqueOperacao,
                     FSestatico,
                     FSfadiga,
                     r,
                     crossover_rate=0.96,
                     mutation_rate=0.04,
                     n_total=100,
                     max_iter=50):

    colunas_principais = [
        'Material',
        'E (GPa)',
        'Método de Fabricação / Tratamento',
        'Limite de Escoamento (MPa)',
        'Limite de Ruptura (MPa)',
        'Diametro S1',
        'Diametro S2',
        'Diametro S3',
        'Diametro S4',
        'Diametro S5',
        'Diametro S6',
        'Comprimento S1',
        'Comprimento S2',
        'Comprimento S3',
        'Comprimento S4',
        'Comprimento S5',
        'Comprimento S6',
        'Massa Específica (kg/m³)',
        'Preço Médio (USD/kg)'
    ]


    #Primeiro check === Avaliação Restritiva, Fitness e Filtro de Individuos Viáveis===
    populacao = df_geracao_inicial.copy()
    populacao = filtrar_por_deflexao(populacao)  #primeiro check de deflexão (para melhorar desempenho)
    populacao = filtrar_por_a_valido(populacao, r)
    populacao = filtrar_por_razao_D_L(populacao)
    populacao = avaliacao_restritiva(populacao, MomentosS3, MomentosS4, TorqueMaxPartida, TorqueOperacao, FSfadiga, r)
    populacao = fitness(populacao)
    populacao_viavel = populacao[populacao['Restrição Atendida'] == "Atende"].copy()

    for iteracao in range(max_iter):

        # === Reposição de indivíduos se necessário ===
        deficit = n_total - len(populacao_viavel)
        while deficit > 0:
            nova_pop = GeracaoInicial(ForcaEntrada, ForcaSaida, dicMateriais, FSestatico)
            nova_pop = avaliacao_restritiva(nova_pop, MomentosS3, MomentosS4, TorqueMaxPartida, TorqueOperacao, FSfadiga, r)
            nova_pop = fitness(nova_pop)
            nova_pop = nova_pop[nova_pop['Restrição Atendida'] == "Atende"]

            # Garante que os DataFrames tenham colunas únicas e bem definidas
            nova_pop = nova_pop.loc[:, ~nova_pop.columns.duplicated()].copy()
            populacao_viavel = populacao_viavel.loc[:, ~populacao_viavel.columns.duplicated()].copy()

            nova_pop.columns = nova_pop.columns.astype(str)
            populacao_viavel.columns = populacao_viavel.columns.astype(str)

            populacao_viavel = pd.concat([populacao_viavel, nova_pop], ignore_index=True)
            deficit = n_total - len(populacao_viavel)

        # === Seleção de pais ===
        n_pais = int(n_total * crossover_rate)
        pais = populacao_viavel.sample(n=n_pais, replace=True).reset_index(drop=True)

        # === Cruzamento: média dos diâmetros + troca de material ===
        filhos = []
        for i in range(0, n_pais, 2):
            if i + 1 >= len(pais): break
            pai1, pai2 = pais.iloc[i], pais.iloc[i + 1]

            col_mat = ['Material','E (GPa)', 'Método de Fabricação / Tratamento',
                       'Limite de Escoamento (MPa)', 'Limite de Ruptura (MPa)',
                       'Preço Médio (USD/kg)']
            col_d = ['Diametro S1', 'Diametro S2', 'Diametro S3',
                     'Diametro S4', 'Diametro S5', 'Diametro S6']

            media_diametros = (pai1[col_d].values + pai2[col_d].values) / 2

            filho1 = pai1.copy()
            filho2 = pai2.copy()
            for j, col in enumerate(col_d):
                filho1[col] = media_diametros[j]
                filho2[col] = media_diametros[j]
            for col in col_mat:
                filho1[col] = pai2[col]
                filho2[col] = pai1[col]

            filhos.extend([filho1, filho2])

        df_filhos = pd.DataFrame(filhos)

        # === Mutação (perturbação aleatória nos diâmetros) ===
        n_mutar = int(len(df_filhos) * mutation_rate)
        idx_mutar = np.random.choice(df_filhos.index, size=n_mutar, replace=False)
        for idx in idx_mutar:
            for d in ['Diametro S1', 'Diametro S2', 'Diametro S3',
                      'Diametro S4', 'Diametro S5', 'Diametro S6']:
                df_filhos.at[idx, d] *= np.random.uniform(0.95, 1.05)

        # === Avaliação dos filhos ===
        df_filhos = df_filhos.filter(items=colunas_principais)
        df_filhos = fitness(df_filhos)

        # === Combinação e seleção dos melhores ===
        combinada = pd.concat([populacao_viavel, df_filhos], ignore_index=True)
        combinada = combinada.sort_values(by='Massa*Custo', ascending=True).head(n_total).reset_index(drop=True)

        # === Seleciona somente as colunas relevantes
        combinada = combinada.filter(items=colunas_principais)

        # Atualiza população
        populacao = combinada.copy()
        populacao = filtrar_por_deflexao(populacao)  #primeiro check de deflexão (para melhorar desempenho)
        populacao = filtrar_por_a_valido(populacao, r)
        populacao = filtrar_por_razao_D_L(populacao)
        populacao = avaliacao_restritiva(populacao, MomentosS3, MomentosS4, TorqueMaxPartida, TorqueOperacao, FSfadiga, r)
        populacao = fitness(populacao)
        populacao = populacao[populacao['Restrição Atendida'] == "Atende"].copy()

    return populacao


primeiro_teste_NSGA2 = nsga2_otimizacao(df_GeracaoInicial, fEntrada, fSaida, dicMateriais, listaMomentosS3, listaMomentosS4, torquePartidaMax, torqueOperaçao, 3, 2, 1)


#GERAÇÃO DE OUTPUTS
for i in range(1, 11):
    df_GeracaoInicial = GeracaoInicial(fEntrada, fSaida, dicMateriais, 3)
    primeiro_teste_NSGA2 = nsga2_otimizacao(df_GeracaoInicial, fEntrada, fSaida, dicMateriais, listaMomentosS3, listaMomentosS4, torquePartidaMax, torqueOperaçao, 3, 2, 1)
    primeiro_teste_NSGA2['Iteração'] = i'
    primeiro_teste_NSGA2.to_csv(rf'sua_paista_saída_teração_{i}.csv', index=False)  #substitua o prefixo sua_paista_saída_" pelo path destino


