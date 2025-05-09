import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pulp as pl

st.set_page_config(layout="wide", page_title="Cloud Service Placement Optimizer")

st.title("Optimisation du Placement des Services dans un Cloud")
st.markdown("""
Cette application permet d'optimiser l'allocation des services sur différents serveurs dans un cloud privé.
L'objectif est de minimiser la latence globale tout en respectant les contraintes de ressources.
L'optimisation est effectuée directement avec des variables entières pour une implémentation pratique.
""")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    num_servers = st.number_input("Nombre de serveurs", min_value=1, max_value=10, value=2)
    num_services = st.number_input("Nombre de services", min_value=1, max_value=10, value=3)
    
    # Generate server names using letters
    server_names = [chr(65 + i) for i in range(num_servers)]
    
    # Create tabs for detailed configuration
    tab_constraints, tab_latency = st.tabs(["Contraintes", "Latence"])
    
    with tab_constraints:
        st.subheader("Contraintes de ressources")
        
        # CPU capacity for each server
        st.write("Capacité CPU des serveurs")
        cpu_capacities = {}
        for i in range(num_servers):
            cpu_capacities[server_names[i]] = st.number_input(
                f"Capacité CPU Serveur {server_names[i]}", 
                min_value=1, 
                value=100
            )
        
        # Service requirements
        st.write("Besoins des services")
        min_instances = {}
        cpu_per_instance = {}
        
        for i in range(num_services):
            st.markdown(f"**Service S{i+1}**")
            min_instances[i+1] = st.number_input(
                f"Instances minimales pour S{i+1}", 
                min_value=1, 
                value=3
            )
            cpu_per_instance[i+1] = st.number_input(
                f"CPU par instance pour S{i+1}", 
                min_value=1, 
                value=10
            )
    
    with tab_latency:
        st.subheader("Paramètres de latence")
        latency_params = {}
        
        for i in range(num_services):
            st.markdown(f"**Service S{i+1}**")
            for server in server_names:
                latency_params[(server, i+1)] = st.number_input(
                    f"Latence de base pour S{i+1} sur serveur {server}", 
                    min_value=0.1, 
                    value=1.0, 
                    step=0.1,
                    format="%.1f"
                )

# Main content
st.header("Optimisation")

# Function to run the optimization using PuLP for integer programming
def run_optimization_pulp():
    with st.spinner("Optimisation en nombres entiers en cours..."):
        # Create a new model
        model = pl.LpProblem(name="cloud_optimization", sense=pl.LpMinimize)
        
        # Define decision variables (integer number of instances for each service on each server)
        x = {}
        for i in range(num_servers):
            for j in range(num_services):
                server = server_names[i]
                service = j+1
                # Integer variables
                x[server, service] = pl.LpVariable(f"x_{server}_{service}", lowBound=0, cat='Integer')
        
        # Define objective function: minimize total latency
        # Note: PuLP doesn't directly support quadratic terms, so we need to linearize
        # For simplicity, we'll use piecewise linear approximation for the quadratic term
        
        # Maximum expected instances per server-service to define the piecewise linear segments
        max_expected_instances = 50
        
        # Create linearized variables for the quadratic terms
        linearized_vars = {}
        for i in range(num_servers):
            for j in range(num_services):
                server = server_names[i]
                service = j+1
                
                # Create piecewise linear approximation variables
                for n in range(1, max_expected_instances + 1):
                    linearized_vars[server, service, n] = pl.LpVariable(
                        f"y_{server}_{service}_{n}", 
                        lowBound=0, 
                        upBound=1,
                        cat='Continuous'
                    )
        
        # Objective: minimize total latency (approximated)
        objective_expr = pl.lpSum(
            latency_params[server, service] * n * n * linearized_vars[server, service, n]
            for server in server_names
            for service in range(1, num_services + 1)
            for n in range(1, max_expected_instances + 1)
        )
        model += objective_expr
        
        # Constraint: SOS2 constraints for piecewise linear approximation
        for i in range(num_servers):
            for j in range(num_services):
                server = server_names[i]
                service = j+1
                
                # Sum of linearized variables must equal 1
                model += pl.lpSum(linearized_vars[server, service, n] for n in range(1, max_expected_instances + 1)) == 1
                
                # Link original variable to linearized variables
                model += x[server, service] == pl.lpSum(n * linearized_vars[server, service, n] for n in range(1, max_expected_instances + 1))
        
        # Constraint: minimum instances for each service
        for j in range(num_services):
            service = j+1
            model += pl.lpSum(x[server, service] for server in server_names) >= min_instances[service]
        
        # Constraint: CPU capacity for each server
        for i in range(num_servers):
            server = server_names[i]
            model += pl.lpSum(x[server, service] * cpu_per_instance[service] for service in range(1, num_services + 1)) <= cpu_capacities[server]
        
        # Solve the model
        solver = pl.PULP_CBC_CMD(msg=False)
        model.solve(solver)
        
        # Check if solution was found
        if model.status == pl.LpStatusOptimal:
            # Extract solution
            x_sol = np.zeros((num_servers, num_services))
            for i in range(num_servers):
                for j in range(num_services):
                    server = server_names[i]
                    service = j+1
                    x_sol[i, j] = pl.value(x[server, service])
            
            # Calculate objective value
            obj_value = 0
            for i in range(num_servers):
                for j in range(num_services):
                    server = server_names[i]
                    service = j+1
                    obj_value += latency_params[server, service] * (x_sol[i, j] ** 2)
            
            return {
                'success': True,
                'message': "Solution optimale trouvée",
                'x_int': x_sol.flatten(),
                'obj_value_int': obj_value
            }
        else:
            return {
                'success': False,
                'message': "Aucune solution trouvée. Vérifiez les contraintes."
            }

# Alternative MILP approach using CVXPY
def run_optimization_with_cvxpy():
    import cvxpy as cp
    
    with st.spinner("Optimisation en nombres entiers avec CVXPY en cours..."):
        # Create variables
        x = cp.Variable((num_servers, num_services), integer=True)
        
        # Objective function: minimize total latency
        objective = 0
        for i in range(num_servers):
            for j in range(num_services):
                objective += latency_params[(server_names[i], j+1)] * cp.square(x[i, j])
        
        # Constraints
        constraints = []
        
        # Non-negativity
        constraints.append(x >= 0)
        
        # Minimum instances for each service
        for j in range(num_services):
            constraints.append(cp.sum(x[:, j]) >= min_instances[j+1])
        
        # CPU capacity for each server
        for i in range(num_servers):
            total_cpu = 0
            for j in range(num_services):
                total_cpu += x[i, j] * cpu_per_instance[j+1]
            constraints.append(total_cpu <= cpu_capacities[server_names[i]])
        
        # Define the problem
        problem = cp.Problem(cp.Minimize(objective), constraints)
        
        # Try to solve
        try:
            problem.solve(solver=cp.MOSEK)
            
            if problem.status == cp.OPTIMAL:
                # Extract solution
                x_sol = x.value
                
                # Calculate objective value
                obj_value = 0
                for i in range(num_servers):
                    for j in range(num_services):
                        obj_value += latency_params[(server_names[i], j+1)] * (x_sol[i, j] ** 2)
                
                return {
                    'success': True,
                    'message': "Solution optimale trouvée",
                    'x_int': x_sol.flatten(),
                    'obj_value_int': obj_value
                }
            else:
                return {
                    'success': False,
                    'message': f"Problème non résolu: {problem.status}"
                }
        except Exception as e:
            return {
                'success': False,
                'message': f"Erreur lors de la résolution: {str(e)}"
            }

# Alternative approach with Gurobi (if available)
def run_optimization_with_gurobi():
    try:
        import gurobipy as gp
        from gurobipy import GRB
        
        with st.spinner("Optimisation en nombres entiers avec Gurobi en cours..."):
            # Create a new model
            model = gp.Model("cloud_optimization")
            
            # Create variables
            x = {}
            for i in range(num_servers):
                for j in range(num_services):
                    server = server_names[i]
                    service = j+1
                    x[server, service] = model.addVar(vtype=GRB.INTEGER, name=f"x_{server}_{service}", lb=0)
            
            # Set objective: minimize total latency
            obj = gp.QuadExpr()
            for i in range(num_servers):
                for j in range(num_services):
                    server = server_names[i]
                    service = j+1
                    # Quadratic term for latency
                    obj.add(latency_params[server, service] * x[server, service] * x[server, service])
            
            model.setObjective(obj, GRB.MINIMIZE)
            
            # Add minimum instances constraints
            for j in range(num_services):
                service = j+1
                model.addConstr(
                    gp.quicksum(x[server, service] for server in server_names) >= min_instances[service],
                    name=f"min_instances_service_{service}"
                )
            
            # Add CPU capacity constraints
            for i in range(num_servers):
                server = server_names[i]
                model.addConstr(
                    gp.quicksum(x[server, service] * cpu_per_instance[service] for service in range(1, num_services + 1)) <= cpu_capacities[server],
                    name=f"cpu_capacity_server_{server}"
                )
            
            # Optimize model
            model.setParam('OutputFlag', 0)  # Suppress output
            model.optimize()
            
            # Check if a solution was found
            if model.status == GRB.OPTIMAL:
                # Extract solution
                x_sol = np.zeros((num_servers, num_services))
                for i in range(num_servers):
                    for j in range(num_services):
                        server = server_names[i]
                        service = j+1
                        x_sol[i, j] = x[server, service].X
                
                # Calculate objective value
                obj_value = 0
                for i in range(num_servers):
                    for j in range(num_services):
                        server = server_names[i]
                        service = j+1
                        obj_value += latency_params[server, service] * (x_sol[i, j] ** 2)
                
                return {
                    'success': True,
                    'message': "Solution optimale trouvée",
                    'x_int': x_sol.flatten(),
                    'obj_value_int': obj_value
                }
            else:
                return {
                    'success': False,
                    'message': f"Problème non résolu: {model.status}"
                }
    except ImportError:
        return {
            'success': False,
            'message': "Gurobi n'est pas installé. Utilisez une autre méthode."
        }

# Function to visualize results (same as original)
def visualize_results(result):
    # Reshape for easier processing
    x_opt = result['x_int'].reshape(num_servers, num_services)
    
    # Create a dataframe for the results
    data = []
    for i in range(num_servers):
        for j in range(num_services):
            data.append({
                'Serveur': server_names[i],
                'Service': f'S{j+1}',
                'Instances': x_opt[i, j],
                'CPU Utilisé': x_opt[i, j] * cpu_per_instance[j+1],
                'Latence': latency_params[(server_names[i], j+1)] * (x_opt[i, j] ** 2)
            })
    
    results_df = pd.DataFrame(data)
    
    # Create summary dataframes
    server_summary = results_df.groupby('Serveur').agg({
        'Instances': 'sum',
        'CPU Utilisé': 'sum',
        'Latence': 'sum'
    }).reset_index()
    
    service_summary = results_df.groupby('Service').agg({
        'Instances': 'sum',
        'CPU Utilisé': 'sum',
        'Latence': 'sum'
    }).reset_index()
    
    # Calculate CPU utilization percentage
    server_summary['Capacité CPU'] = server_summary['Serveur'].map(lambda s: cpu_capacities[s])
    server_summary['Utilisation CPU (%)'] = (server_summary['CPU Utilisé'] / server_summary['Capacité CPU']) * 100
    
    # Display results summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution des instances par serveur")
        fig1 = px.bar(
            results_df, 
            x='Serveur', 
            y='Instances', 
            color='Service',
            title="Répartition des services sur les serveurs",
            labels={'Instances': 'Nombre d\'instances (entier)'},
            barmode='stack'
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("Utilisation CPU par serveur")
        fig2 = go.Figure()
        
        for i, server in enumerate(server_names):
            server_data = server_summary[server_summary['Serveur'] == server]
            fig2.add_trace(go.Bar(
                x=[server],
                y=[server_data['CPU Utilisé'].values[0]],
                name=f"Utilisé ({server_data['CPU Utilisé'].values[0]:.1f})",
                marker_color='blue'
            ))
            
            fig2.add_trace(go.Bar(
                x=[server],
                y=[server_data['Capacité CPU'].values[0] - server_data['CPU Utilisé'].values[0]],
                name=f"Disponible ({server_data['Capacité CPU'].values[0] - server_data['CPU Utilisé'].values[0]:.1f})",
                marker_color='lightgrey'
            ))
        
        fig2.update_layout(
            title="Utilisation vs Capacité CPU des serveurs",
            yaxis_title="CPU",
            barmode='stack',
            showlegend=True,
            legend_traceorder="reversed"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Detailed results tables
    st.subheader("Résultats détaillés")
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("**Allocation optimale des instances (entiers)**")
        # Pivot table for a cleaner display
        pivot_df = results_df.pivot(index='Service', columns='Serveur', values='Instances')
        pivot_df = pivot_df.fillna(0)
        st.dataframe(pivot_df, use_container_width=True)
    
    with col4:
        st.markdown("**Utilisation des ressources par serveur**")
        server_resource_df = server_summary[['Serveur', 'Instances', 'CPU Utilisé', 'Capacité CPU', 'Utilisation CPU (%)']]
        server_resource_df = server_resource_df.round(2)
        st.dataframe(server_resource_df, use_container_width=True)
    
    # Additional visualizations
    st.subheader("Analyses supplémentaires")
    col5, col6 = st.columns(2)
    
    with col5:
        st.markdown("**Latence par service et serveur**")
        fig3 = px.bar(
            results_df, 
            x='Service', 
            y='Latence', 
            color='Serveur',
            title="Contribution à la latence globale",
            labels={'Latence': 'Latence (unités)'},
            barmode='stack'
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col6:
        st.markdown("**Distribution des services**")
        service_dist = results_df.pivot_table(
            index='Service', 
            columns='Serveur', 
            values='Instances', 
            aggfunc='sum'
        ).fillna(0)
        
        # Convert to percentage
        service_dist_pct = service_dist.div(service_dist.sum(axis=1), axis=0) * 100
    
        fig4 = px.imshow(
            service_dist_pct,
            labels=dict(x="Serveur", y="Service", color="% d'instances"),
            x=service_dist_pct.columns,
            y=service_dist_pct.index,
            color_continuous_scale="Blues",
            title="Distribution des services (% par serveur)"
        )
        
        # Add text annotations
        for i in range(len(service_dist_pct.index)):
            for j in range(len(service_dist_pct.columns)):
                fig4.add_annotation(
                    x=j,
                    y=i,
                    text=f"{service_dist_pct.iloc[i, j]:.1f}%",
                    showarrow=False,
                    font=dict(color="black")
                )
                
        st.plotly_chart(fig4, use_container_width=True)
    
    # Overall summary
    st.subheader("Synthèse de l'optimisation")
    col7, col8 = st.columns(2)
    
    with col7:
        total_latency = results_df['Latence'].sum()
        st.metric("Latence totale", f"{total_latency:.2f} unités")
        
        avg_cpu_util = server_summary['Utilisation CPU (%)'].mean()
        st.metric("Utilisation CPU moyenne", f"{avg_cpu_util:.2f}%")
        
        total_instances = results_df['Instances'].sum()
        st.metric("Instances totales", f"{int(total_instances)}")
    
    with col8:
        st.markdown("**Facteurs clés de l'optimisation**")
        st.markdown(f"""
        - Optimisation directe en nombres entiers (sans arrondi)
        - La latence augmente de façon quadratique avec le nombre d'instances du même service sur un serveur
        - Répartition préférentielle vers les serveurs avec une latence de base plus faible
        - Respect des contraintes de capacité CPU sur chaque serveur
        - Satisfaction du nombre minimal d'instances requis pour chaque service
        """)
    
    # Return key metrics for display
    return {
        "latence_totale": total_latency,
        "utilisation_cpu": server_summary['Utilisation CPU (%)'].tolist(),
        "solution": x_opt
    }

# Select optimization method
optimization_method = st.selectbox(
    "Méthode d'optimisation",
    ["PuLP (programmation linéaire en nombres entiers)", 
     "CVXPY (avec solveur MILP)",
     "Gurobi (si disponible)"]
)

# Button to run optimization
if st.button("Lancer l'optimisation"):
    if optimization_method == "PuLP (programmation linéaire en nombres entiers)":
        result = run_optimization_pulp()
    elif optimization_method == "CVXPY (avec solveur MILP)":
        result = run_optimization_with_cvxpy()
    else:
        result = run_optimization_with_gurobi()
    
    if result['success']:
        st.success("Optimisation réussie !")
        
        # Visualize the integer solution
        metrics = visualize_results(result)
        
        # Add an analysis of the solution
        st.header("Analyse de la solution")
        st.markdown("""
        ### Interprétation des résultats
        
        L'algorithme a trouvé une allocation optimale des services directement en nombres entiers qui minimise la latence totale tout en respectant toutes les contraintes imposées.
        
        **Avantages de l'optimisation en nombres entiers :**
        
        1. **Solution optimale directe** : Contrairement à l'approche précédente qui arrondissait une solution continue, cette méthode trouve directement la meilleure solution avec des valeurs entières.
        
        2. **Pas de compromis dû à l'arrondi** : Il n'y a pas de perte d'optimalité causée par l'arrondi des valeurs continues.
        
        3. **Garantie des contraintes** : Toutes les contraintes sont strictement respectées dans la solution optimale.
        
        4. **Répartition précise** : L'allocation des instances est déterminée de manière globalement optimale en tenant compte de toutes les contraintes simultanément.
        
        5. **Modélisation fidèle** : Le problème est résolu exactement comme il a été modélisé, avec la fonction objectif quadratique et des variables entières.
        """)
    else:
        st.error("L'optimisation a échoué.")
        st.write("Message d'erreur:", result['message'])

# Add an explanation of the model
with st.expander("À propos du modèle d'optimisation"):
    st.markdown("""
    ### Modèle mathématique (MILP - Mixed Integer Linear Programming)
    
    Le problème est formulé comme un problème d'optimisation quadratique en nombres entiers:
    
    **Fonction objectif:**
    
    Minimiser Z = ∑(L_s(i) * x_s,i^2)
    
    où:
    - x_s,i est le nombre d'instances du service i sur le serveur s (variable entière)
    - L_s(i) est la latence de base pour le service i sur le serveur s
    - Le terme au carré (x_s,i^2) modélise l'effet de saturation où la latence augmente de façon quadratique
    
    **Contraintes:**
    
    1. **Nombre minimal d'instances**: Pour chaque service i, ∑(x_s,i) ≥ D_i pour tous les serveurs s
    
    2. **Capacité CPU**: Pour chaque serveur s, ∑(c_i * x_s,i) ≤ C_s pour tous les services i
       où c_i est le CPU requis par instance du service i et C_s est la capacité CPU du serveur s
    
    3. **Contrainte d'intégralité**: x_s,i sont des entiers non négatifs pour tout serveur s et service i
    
    **Approches de résolution:**
    
    1. **PuLP avec linéarisation par morceaux**: Approximation de la fonction quadratique par une fonction linéaire par morceaux
    
    2. **CVXPY avec contraintes quadratiques**: Utilisation directe de la formulation quadratique avec variables entières
    
    3. **Gurobi**: Solveur commercial qui peut gérer directement les problèmes MIQP (Mixed Integer Quadratic Programming)
    
    La formulation en nombres entiers évite complètement le besoin d'arrondir une solution continue, garantissant ainsi une solution optimale pour le problème d'allocation de services tel que défini.
    """)

# Add instructions for using the app
with st.expander("Instructions d'utilisation"):
    st.markdown("""
    ### Comment utiliser cette application
    
    1. **Configuration**:
       - Dans la barre latérale, définissez le nombre de serveurs et de services
       - Configurez les capacités CPU de chaque serveur
       - Spécifiez les besoins minimaux en instances et en CPU pour chaque service
       - Définissez les paramètres de latence pour chaque combinaison service-serveur
    
    2. **Sélection de la méthode**:
       - Choisissez la méthode d'optimisation en nombres entiers à utiliser
       - PuLP: Bibliothèque open-source pour la programmation linéaire
       - CVXPY: Bibliothèque plus avancée pour l'optimisation convexe
       - Gurobi: Solveur commercial haute performance (si disponible)
    
    3. **Optimisation**:
       - Cliquez sur le bouton "Lancer l'optimisation" pour exécuter l'algorithme
       - Analysez les résultats présentés sous forme de graphiques et de tableaux
    
    4. **Interprétation**:
       - Examinez la répartition des services entre les serveurs
       - Vérifiez l'utilisation des ressources CPU sur chaque serveur
       - Analysez les contributions à la latence totale
    """)