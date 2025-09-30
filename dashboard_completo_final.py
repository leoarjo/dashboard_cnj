import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings
import psycopg2 
from psycopg2 import sql 
from dotenv import load_dotenv
import os
warnings.filterwarnings('ignore')

load_dotenv()  # Carrega as variáveis do .env

# --- CONFIGURAÇÕES DO BANCO DE DADOS ---
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
# ---------------------------------------

# Configuração da página
st.set_page_config(
    page_title="Metas CNJ | Projeto Integrador",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f4e79;
    }
    .data-source {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background: #f0f8e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Classe para modelos preditivos
class ModeloPredicaoMetas:
    def __init__(self, df):
        self.df = df.copy()
        self.modelo_iad = None
        self.modelo_cumprimento = None

    def preparar_dados(self):
        """Preparar dados para modelagem"""
        # Features para predição
        features = [
            'processos_distribuidos',
            'processos_pendentes',
            'taxa_congestionamento', 
            'tempo_medio_tramitacao',
            'processos_antigos_pendentes',
            'mes'
        ]

        # Criar features adicionais
        self.df['processos_julgados'] = pd.to_numeric(self.df['processos_julgados'], errors='coerce')
        self.df['processos_distribuidos'] = pd.to_numeric(self.df['processos_distribuidos'], errors='coerce')
        self.df['processos_pendentes'] = pd.to_numeric(self.df['processos_pendentes'], errors='coerce')
        
        self.df['razao_julgados_distribuidos'] = self.df['processos_julgados'] / self.df['processos_distribuidos']
        self.df['densidade_processos'] = self.df['processos_pendentes'] / self.df['processos_julgados']

        features.extend(['razao_julgados_distribuidos', 'densidade_processos'])

        # Remover NaN e infinitos
        self.df = self.df.replace([np.inf, -np.inf], np.nan).dropna(subset=features)

        return features

    def treinar_modelos(self):
        """Treinar ambos os modelos"""
        features = self.preparar_dados()

        if len(self.df) < 10:
            return False, "Dados insuficientes para treinamento"

        X = self.df[features]
        y_iad = self.df['iad_meta1']
        y_cumprimento = (self.df['iad_meta1'] >= 1.0).astype(int)

        # Dividir dados
        X_train, X_test, y_iad_train, y_iad_test = train_test_split(
            X, y_iad, test_size=0.3, random_state=42
        )
        _, _, y_cum_train, y_cum_test = train_test_split(
            X, y_cumprimento, test_size=0.3, random_state=42
        )

        # Treinar modelo de IAD
        self.modelo_iad = RandomForestRegressor(n_estimators=50, random_state=42)
        self.modelo_iad.fit(X_train, y_iad_train)

        # Treinar modelo de cumprimento
        self.modelo_cumprimento = LogisticRegression(random_state=42, max_iter=500)
        self.modelo_cumprimento.fit(X_train, y_cum_train)

        # Avaliar modelos
        y_iad_pred = self.modelo_iad.predict(X_test)
        y_cum_pred = self.modelo_cumprimento.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_iad_test, y_iad_pred))
        acc = accuracy_score(y_cum_test, y_cum_pred)

        return True, {
            'rmse': rmse,
            'accuracy': acc,
            'features': features
        }

    def simular_cenarios(self):
        """Simular diferentes cenários"""
        if self.modelo_iad is None or self.modelo_cumprimento is None:
            return None

        features = self.preparar_dados()
        cenario_base = self.df[features].mean().values.reshape(1, -1)

        # Cenários
        cenario_otimista = cenario_base.copy()
        cenario_otimista[0][2] *= 0.8  # reduzir congestionamento 20%
        cenario_otimista[0][3] *= 0.85  # reduzir tempo 15%

        cenario_pessimista = cenario_base.copy()
        cenario_pessimista[0][2] *= 1.15  # aumentar congestionamento 15%
        cenario_pessimista[0][3] *= 1.2  # aumentar tempo 20%

        # Predições
        iad_base = self.modelo_iad.predict(cenario_base)[0]
        iad_otimista = self.modelo_iad.predict(cenario_otimista)[0]
        iad_pessimista = self.modelo_iad.predict(cenario_pessimista)[0]

        prob_base = self.modelo_cumprimento.predict_proba(cenario_base)[0][1]
        prob_otimista = self.modelo_cumprimento.predict_proba(cenario_otimista)[0][1]
        prob_pessimista = self.modelo_cumprimento.predict_proba(cenario_pessimista)[0][1]

        return {
            'base': {'iad': iad_base, 'prob': prob_base},
            'otimista': {'iad': iad_otimista, 'prob': prob_otimista},
            'pessimista': {'iad': iad_pessimista, 'prob': prob_pessimista}
        }

# Função para carregar dados do BD
@st.cache_data(ttl=600) # Armazena em cache por 10 minutos
def carregar_dados():
    """Tenta carregar dados agregados da tabela metas_cnj do PostgreSQL."""
    try:
        # 1. Conectar ao banco de dados
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        
        # 2. Query para selecionar todos os dados agregados
        query = "SELECT * FROM metas_cnj ORDER BY tribunal, ano, mes;"
        
        # 3. Usar pandas para ler o resultado diretamente
        df = pd.read_sql(query, conn)
        
        # 4. Fechar conexão
        conn.close()
        
        # 5. Tratamento de dados no DataFrame (Crucial para visualização e ML)
        df['mes'] = pd.to_numeric(df['mes'], errors='coerce')
        df['ano'] = pd.to_numeric(df['ano'], errors='coerce')
        
        # Criar a coluna 'data' para a evolução temporal
        df['data'] = pd.to_datetime(df['ano'].astype(str) + '-' + df['mes'].astype(str) + '-01', errors='coerce')

        # Remove linhas com valores NaN que surgiram do parse (ex: datas inválidas)
        df = df.dropna(subset=['data', 'mes', 'ano', 'processos_julgados', 'processos_distribuidos']).copy()
        
        # 6. Fonte dos dados
        fonte_dados = f"Dados ATUALIZADOS do PostgreSQL ({DB_HOST}, Tabela metas_cnj)"

        return df, fonte_dados

    except Exception as e:
        # Fallback para dados locais (se falhar ao conectar)
        st.error(f"❌ Falha ao conectar ao banco de dados: {e}")
        try:
            # Tenta carregar dados de um CSV de fallback
            df = pd.read_csv('dados_finais_projeto_integrador.csv')
            df['data'] = pd.to_datetime(df['data'])
            return df, "Dados LOCAIS (CSV de Fallback)"
        except FileNotFoundError:
             st.error("❌ Nenhum arquivo de dados local encontrado!")
             return None, None
# --- FIM DA FUNÇÃO DE CARREGAMENTO ---

# Função principal
def main():
    # Título principal
    st.markdown('<h1 class="main-header">⚖️ Metas CNJ | Projeto Integrador IESB</h1>', unsafe_allow_html=True)
    st.markdown("### 📊 Análises Exploratórias + Modelos Preditivos + Relatórios Interativos")

    # Carregar dados
    df, fonte_dados = carregar_dados()
    if df is None:
        return

    # Informação sobre fonte
    st.markdown(f"""
    <div class="data-source">
        <h4>🔗 Fonte dos Dados</h4>
        <p><strong>{fonte_dados}</strong></p>
        <p>• API Pública DataJud: https://api-publica.datajud.cnj.jus.br/<br>
        • Período: {df['data'].min().strftime('%B/%Y')} a {df['data'].max().strftime('%B/%Y')}<br>
        • Tribunais: {len(df['tribunal'].unique())} Regionais do Trabalho (TRTs)<br>
        • Registros (Agregados): {len(df)} observações</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("🔍 Controles do Dashboard")
    st.sidebar.markdown("---")

    # Seleção de funcionalidades
    opcao_analise = st.sidebar.selectbox(
        "Selecione o Tipo de Análise:",
        ["📊 Análises Exploratórias", "🤖 Modelos Preditivos", "📋 Relatórios Interativos", "🔄 Visão Completa"]
    )

    # 1. Obtém todos os TRTs disponíveis
    todos_tribunais_disponiveis = sorted(df['tribunal'].unique())
    
    # 2. Define o valor padrão para INCLUIR TODOS OS TRTS DISPONÍVEIS
    default_tribunais = todos_tribunais_disponiveis 

    # Filtros comuns
    tribunais_selecionados = st.sidebar.multiselect(
        "Tribunais:",
        options=todos_tribunais_disponiveis,
        default=default_tribunais # <--- AGORA SELECIONA TUDO POR PADRÃO
    )

    meses_selecionados = st.sidebar.multiselect(
        "Meses:",
        options=sorted(df['mes'].unique()),
        default=sorted(df['mes'].unique())
    )

    # Aplicar filtros
    if tribunais_selecionados and meses_selecionados:
        df_filtrado = df[
            (df['tribunal'].isin(tribunais_selecionados)) & 
            (df['mes'].isin(meses_selecionados))
        ]
    else:
        df_filtrado = df

    if df_filtrado.empty:
        st.warning("⚠️ Nenhum dado encontrado com os filtros selecionados.")
        return

    # SEÇÃO: MÉTRICAS PRINCIPAIS (sempre visível)
    st.header("📈 Indicadores Consolidados")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_distribuidos = df_filtrado['processos_distribuidos'].sum()
        st.metric("📥 Distribuídos", f"{total_distribuidos:,}".replace(',', '.'))

    with col2:
        total_julgados = df_filtrado['processos_julgados'].sum()
        delta = total_julgados - total_distribuidos
        st.metric("⚖️ Julgados", f"{total_julgados:,}".replace(',', '.'), 
                    delta=f"{delta:+,}".replace(',', '.'))

    with col3:
        iad_medio = df_filtrado['iad_meta1'].mean()
        status = "✅ Cumprindo" if iad_medio >= 1.0 else "❌ Não cumprindo"
        st.metric("🎯 IAD Médio", f"{iad_medio:.3f}", delta=status)

    with col4:
        congestionamento = df_filtrado['taxa_congestionamento'].mean()
        st.metric("🚦 Congestionamento", f"{congestionamento:.1f}%")

    st.markdown("---")

    # RENDERIZAÇÃO BASEADA NA OPÇÃO SELECIONADA
    if opcao_analise == "📊 Análises Exploratórias" or opcao_analise == "🔄 Visão Completa":
        st.header("📊 Análises Exploratórias")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 IAD por Tribunal")
            iad_tribunal = df_filtrado.groupby('tribunal')['iad_meta1'].mean().sort_values(ascending=False)

            fig_iad = px.bar(
                x=iad_tribunal.index,
                y=iad_tribunal.values,
                title="Índice de Atendimento à Demanda",
                color=iad_tribunal.values,
                color_continuous_scale='RdYlGn',
                color_continuous_midpoint=1.0
            )
            fig_iad.add_hline(y=1.0, line_dash="dash", line_color="blue", 
                              annotation_text="Meta CNJ: 1.0")
            fig_iad.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_iad, use_container_width=True)

        with col2:
            st.subheader("📈 Evolução Temporal")
            evolucao = df_filtrado.groupby('data').agg({
                'processos_distribuidos': 'sum',
                'processos_julgados': 'sum'
            }).reset_index()

            fig_evolucao = go.Figure()
            fig_evolucao.add_trace(go.Scatter(
                x=evolucao['data'], y=evolucao['processos_distribuidos'],
                mode='lines+markers', name='Distribuídos', line=dict(color='blue')
            ))
            fig_evolucao.add_trace(go.Scatter(
                x=evolucao['data'], y=evolucao['processos_julgados'],
                mode='lines+markers', name='Julgados', line=dict(color='green')
            ))
            fig_evolucao.update_layout(title="Evolução Mensal", height=400)
            st.plotly_chart(fig_evolucao, use_container_width=True)

        # Tabela de ranking
        st.subheader("🏆 Ranking de Performance")
        ranking = df_filtrado.groupby('tribunal').agg({
            'processos_distribuidos': 'sum',
            'processos_julgados': 'sum',
            'iad_meta1': 'mean',
            'taxa_congestionamento': 'mean'
        }).round(3).sort_values('iad_meta1', ascending=False)

        ranking.columns = ['Distribuídos', 'Julgados', 'IAD', 'Congestionamento (%)']
        st.dataframe(ranking, use_container_width=True)

    if opcao_analise == "🤖 Modelos Preditivos" or opcao_analise == "🔄 Visão Completa":
        st.header("🤖 Análises Preditivas")

        # Treinar modelos
        with st.spinner("Treinando modelos de machine learning..."):
            modelo = ModeloPredicaoMetas(df_filtrado)
            sucesso, resultado = modelo.treinar_modelos()

        if sucesso:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                <div class="prediction-box">
                    <h4>🎯 Performance dos Modelos</h4>
                    <p><strong>Modelo de Predição do IAD:</strong><br>
                    RMSE: {:.4f}</p>
                    <p><strong>Modelo de Cumprimento:</strong><br>
                    Acurácia: {:.1%}</p>
                </div>
                """.format(resultado['rmse'], resultado['accuracy']), unsafe_allow_html=True)

            with col2:
                st.subheader("🔮 Simulação de Cenários")
                cenarios = modelo.simular_cenarios()

                if cenarios:
                    st.markdown("**Cenário Base (Atual):**")
                    st.write(f"• IAD: {cenarios['base']['iad']:.3f}")
                    st.write(f"• Prob. Cumprimento: {cenarios['base']['prob']:.1%}")

                    st.markdown("**Cenário Otimista (Robô Implementado):**")
                    st.success(f"• IAD: {cenarios['otimista']['iad']:.3f} (+{((cenarios['otimista']['iad'] - cenarios['base']['iad'])/cenarios['base']['iad']*100):+.1f}%)")
                    st.success(f"• Prob. Cumprimento: {cenarios['otimista']['prob']:.1%} (+{(cenarios['otimista']['prob'] - cenarios['base']['prob'])*100:+.1f}pp)")

                    st.markdown("**Cenário Pessimista:**")
                    st.error(f"• IAD: {cenarios['pessimista']['iad']:.3f} ({((cenarios['pessimista']['iad'] - cenarios['base']['iad'])/cenarios['base']['iad']*100):+.1f}%)")
                    st.error(f"• Prob. Cumprimento: {cenarios['pessimista']['prob']:.1%} ({(cenarios['pessimista']['prob'] - cenarios['base']['prob'])*100:+.1f}pp)")
        else:
            st.error(f"❌ Erro no treinamento: {resultado}")

    if opcao_analise == "📋 Relatórios Interativos" or opcao_analise == "🔄 Visão Completa":
        st.header("📋 Relatórios Interativos")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔍 Insights Automáticos")

            # Gerar insights
            iad_geral = df_filtrado['iad_meta1'].mean()
            if iad_geral >= 1.0:
                st.success(f"✅ Meta 1 CUMPRIDA: IAD geral de {iad_geral:.3f}")
            else:
                deficit = (1.0 - iad_geral) * 100
                st.warning(f"⚠️ Meta 1 não cumprida: Déficit de {deficit:.1f}%")

            # Tribunais que precisam atenção
            tribunais_criticos = df_filtrado.groupby('tribunal')['iad_meta1'].mean()
            criticos = tribunais_criticos[tribunais_criticos < 1.0].sort_values()

            if not criticos.empty:
                st.subheader("⚠️ Tribunais que Necessitam Atenção")
                for tribunal, iad in criticos.head(5).items():
                    st.write(f"• **{tribunal}**: IAD {iad:.3f}")

        with col2:
            st.subheader("🚀 Recomendações Específicas")

            st.markdown("""
            **Implementação do Robô Orientador:**

            1. **Prioridade Alta**: Tribunais com IAD < 0.95
            2. **Validação Automática**: Checagens no PJe
            3. **Monitoramento Real-time**: Dashboard permanente
            4. **Capacitação Dirigida**: Treinamento específico
            5. **Alertas Preditivos**: Sistema de avisos
            """)

            # Botão de exportação
            if st.button("📥 Exportar Relatório"):
                csv = df_filtrado.to_csv(index=False)
                st.download_button(
                    label="Baixar CSV",
                    data=csv,
                    file_name=f"relatorio_metas_cnj_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>Projeto Integrador IESB - 2025</strong></p>
        <p>Leonardo Araujo Pereira | Prof. Simone de Araújo Góes Assis</p>
        <p>Fonte: API DataJud/CNJ | Machine Learning: Random Forest + Logistic Regression</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()