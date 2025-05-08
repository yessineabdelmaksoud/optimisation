import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import math

st.set_page_config(layout="wide", page_title="Cloud Service Placement Optimizer")

st.title("Optimisation du Placement des Services dans un Cloud")
st.markdown("""
Cette application permet d'optimiser l'allocation des services sur différents serveurs dans un cloud privé.
L'objectif est de minimiser la latence globale tout en respectant les contraintes de ressources.
Les solutions sont arrondies aux nombres entiers les plus proches pour une implémentation pratique.
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

# Function to create the objective function dynamically
def create_objective_function(num_servers, num_services, latency_params, server_names):
    def objective(x):
        total_latency = 0
        # Reshape flat x array into a matrix where rows are servers and columns are services
        x_matrix = x.reshape(num_servers, num_services)
        
        for i in range(num_servers):
            for j in range(num_services):
                # Use server name (A, B, C, etc.) to access latency parameters
                server = server_names[i]
                service = j+1
                total_latency += latency_params[(server, service)] * (x_matrix[i, j] ** 2)
        
        return total_latency
    
    return objective

# Function to create constraints dynamically
def create_constraints(num_servers, num_services, min_instances, cpu_per_instance, cpu_capacities, server_names):
    constraints = []
    
    # Minimum instances constraint for each service
    for j in range(num_services):
        def min_instance_constraint(x, j=j):
            x_matrix = x.reshape(num_servers, num_services)
            # Sum instances across all servers for this service
            return sum(x_matrix[i, j] for i in range(num_servers)) - min_instances[j+1]
        
        constraints.append({'type': 'ineq', 'fun': min_instance_constraint})
    
    # CPU capacity constraint for each server
    for i in range(num_servers):
        def cpu_constraint(x, i=i):
            x_matrix = x.reshape(num_servers, num_services)
            # Calculate total CPU used on this server
            total_cpu = sum(x_matrix[i, j] * cpu_per_instance[j+1] for j in range(num_services))
            # Constraint is satisfied if CPU used <= capacity
            return cpu_capacities[server_names[i]] - total_cpu
        
        constraints.append({'type': 'ineq', 'fun': cpu_constraint})
    
    return constraints

# Initial guess for optimization
def create_initial_guess(num_servers, num_services, min_instances):
    # Distribute minimum instances equally among servers as a starting point
    x0 = np.zeros((num_servers, num_services))
    for j in range(num_services):
        per_server = min_instances[j+1] / num_servers
        for i in range(num_servers):
            x0[i, j] = per_server
    
    return x0.flatten()

# Function to round the solution to integers while preserving constraints
def round_solution(x_opt, num_servers, num_services, min_instances, cpu_per_instance, cpu_capacities, server_names):
    # Reshape for easier processing
    x_matrix = x_opt.reshape(num_servers, num_services)
    rounded_matrix = np.zeros((num_servers, num_services))
    
    # First, apply floor to all values to get the minimum integer allocations
    for i in range(num_servers):
        for j in range(num_services):
            rounded_matrix[i, j] = math.floor(x_matrix[i, j])
    
    # Check which services need additional instances to meet minimum requirements
    for j in range(num_services):
        service_total = sum(rounded_matrix[i, j] for i in range(num_servers))
        needed = min_instances[j+1] - service_total
        
        if needed > 0:
            # Add needed instances to servers with lowest latency first
            server_latencies = [(i, latency_params[(server_names[i], j+1)]) for i in range(num_servers)]
            server_latencies.sort(key=lambda x: x[1])  # Sort by latency
            
            for server_idx, _ in server_latencies:
                # Calculate available CPU on this server
                current_cpu_used = sum(rounded_matrix[server_idx, s] * cpu_per_instance[s+1] for s in range(num_services))
                available_cpu = cpu_capacities[server_names[server_idx]] - current_cpu_used
                
                # How many instances can we add with available CPU?
                can_add = min(needed, int(available_cpu / cpu_per_instance[j+1]))
                
                if can_add > 0:
                    rounded_matrix[server_idx, j] += can_add
                    needed -= can_add
                
                if needed == 0:
                    break
            
            # If we still need instances but are constrained by CPU, prioritize by latency
            if needed > 0:
                st.warning(f"Impossible de satisfaire toutes les contraintes avec des valeurs entières pour le service S{j+1}.")
    
    return rounded_matrix.flatten()

# Function to check if constraints are satisfied with the rounded solution
def check_constraints(x, num_servers, num_services, min_instances, cpu_per_instance, cpu_capacities, server_names):
    x_matrix = x.reshape(num_servers, num_services)
    violations = []
    
    # Check minimum instances
    for j in range(num_services):
        service_total = sum(x_matrix[i, j] for i in range(num_servers))
        if service_total < min_instances[j+1]:
            violations.append(f"Service S{j+1} a {service_total} instances, besoin de {min_instances[j+1]} minimum.")
    
    # Check CPU capacity
    for i in range(num_servers):
        total_cpu = sum(x_matrix[i, j] * cpu_per_instance[j+1] for j in range(num_services))
        if total_cpu > cpu_capacities[server_names[i]]:
            violations.append(f"Serveur {server_names[i]} utilise {total_cpu} CPU, capacité max {cpu_capacities[server_names[i]]}.")
    
    return violations

# Function to run the optimization
def run_optimization():
    with st.spinner("Optimisation en cours..."):
        # Create objective function and constraints
        objective = create_objective_function(num_servers, num_services, latency_params, server_names)
        constraints = create_constraints(num_servers, num_services, min_instances, cpu_per_instance, cpu_capacities, server_names)
        
        # Initial guess
        x0 = create_initial_guess(num_servers, num_services, min_instances)
        
        # Bounds: all variables must be non-negative
        bounds = [(0, None) for _ in range(num_servers * num_services)]
        
        # Run optimization to get continuous solution
        result = minimize(
            objective, 
            x0, 
            method='SLSQP',  # Sequential Least Squares Programming
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'disp': False}
        )
        
        if result.success:
            # Round solution to integers
            x_int = round_solution(
                result.x, 
                num_servers, 
                num_services, 
                min_instances, 
                cpu_per_instance, 
                cpu_capacities, 
                server_names
            )
            
            # Check constraints for integer solution
            violations = check_constraints(
                x_int, 
                num_servers, 
                num_services, 
                min_instances, 
                cpu_per_instance, 
                cpu_capacities, 
                server_names
            )
            
            # Calculate objective value for integer solution
            obj_value_int = objective(x_int)
            
            return {
                'success': result.success,
                'message': result.message,
                'nit': result.nit,
                'fun': result.fun,
                'x': result.x,  # Original continuous solution
                'x_int': x_int,  # Integer solution
                'obj_value_int': obj_value_int,  # Objective value for integer solution
                'violations': violations  # Any constraint violations
            }
        else:
            return {
                'success': False,
                'message': result.message
            }

# Function to visualize results
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
        - La latence augmente de façon quadratique avec le nombre d'instances du même service sur un serveur
        - Répartition préférentielle vers les serveurs avec une latence de base plus faible
        - Respect des contraintes de capacité CPU sur chaque serveur
        - Satisfaction du nombre minimal d'instances requis pour chaque service
        - Les valeurs sont arrondies aux entiers pour une mise en œuvre pratique
        """)
    
    # Return key metrics for display
    return {
        "latence_totale": total_latency,
        "utilisation_cpu": server_summary['Utilisation CPU (%)'].tolist(),
        "solution": x_opt
    }

# Compare original and integer solutions
def compare_solutions(result):
    st.subheader("Comparaison des solutions continue et entière")
    
    col1, col2 = st.columns(2)
    
    
    
    with col1:
        st.markdown("**Solution en nombres entiers**")
        x_int = result['x_int'].reshape(num_servers, num_services)
        
        int_data = []
        for i in range(num_servers):
            for j in range(num_services):
                int_data.append({
                    'Serveur': server_names[i],
                    'Service': f'S{j+1}',
                    'Instances': int(x_int[i, j])
                })
        
        int_df = pd.DataFrame(int_data)
        int_pivot = int_df.pivot(index='Service', columns='Serveur', values='Instances')
        int_pivot = int_pivot.fillna(0)
        
        st.dataframe(int_pivot, use_container_width=True)
        st.info(f"Latence totale: {result['obj_value_int']:.2f}")
    
    # Show error due to integer rounding
    error_pct = ((result['obj_value_int'] - result['fun']) / result['fun']) * 100
    st.info(f"Écart entre les solutions: {error_pct:.2f}% (différence de latence due à l'arrondi)")

# Button to run optimization
if st.button("Lancer l'optimisation"):
    result = run_optimization()
    
    if result['success']:
        st.success("Optimisation réussie !")
        
        # Show any constraint violations
        if result['violations']:
            st.warning("Attention: La solution entière présente les violations suivantes:")
            for v in result['violations']:
                st.warning(v)
        
        # Compare continuous and integer solutions
        compare_solutions(result)
        
        # Visualize the integer solution
        metrics = visualize_results(result)
        
        # Add an analysis of the solution
        st.header("Analyse de la solution")
        st.markdown("""
        ### Interprétation des résultats
        
        L'algorithme a trouvé une allocation optimale des services qui minimise la latence totale tout en respectant toutes les contraintes imposées. Les valeurs ont été arrondies aux entiers pour une implémentation pratique.
        
        **Observations clés :**
        
        1. **Répartition des services** : Les services sont distribués de manière à équilibrer la charge entre les serveurs, en tenant compte des contraintes de ressources et des paramètres de latence spécifiques à chaque combinaison service-serveur.
        
        2. **Utilisation des ressources** : L'allocation prend en compte l'utilisation du CPU, en évitant de surcharger certains serveurs tout en maximisant l'utilisation des ressources disponibles.
        
        3. **Compromis latence-ressources** : La solution trouve un équilibre entre la minimisation de la latence et l'utilisation efficace des ressources disponibles.
        
        4. **Impact de la latence quadratique** : Le modèle prend en compte l'effet de saturation où la latence augmente de façon quadratique avec le nombre d'instances du même service, ce qui favorise la distribution des services entre les serveurs.
        
        5. **Solutions entières** : Les valeurs fractionnaires ont été arrondies aux entiers les plus proches tout en préservant les contraintes, ce qui peut entraîner une légère augmentation de la latence totale par rapport à la solution mathématique optimale.
        """)
        
        # Show convergence info
        st.subheader("Informations sur la convergence")
        conv_col1, conv_col2 = st.columns(2)
        with conv_col1:
            st.info(f"Nombre d'itérations: {result['nit']}")
            st.info(f"Statut: {result['message']}")
        with conv_col2:
            st.info(f"Fonction objectif continue (latence): {result['fun']:.4f}")
            st.info(f"Fonction objectif entière (latence): {result['obj_value_int']:.4f}")
    else:
        st.error("L'optimisation a échoué. Essayez de modifier les paramètres.")
        st.write("Message d'erreur:", result['message'])

# Add an explanation of the model
with st.expander("À propos du modèle d'optimisation"):
    st.markdown("""
    ### Modèle mathématique
    
    Le problème est formulé comme un problème d'optimisation non linéaire:
    
    **Fonction objectif:**
    
    Minimiser Z = ∑(L_s(i) * x_s,i^2)
    
    où:
    - x_s,i est le nombre d'instances du service i sur le serveur s
    - L_s(i) est la latence de base pour le service i sur le serveur s
    - Le terme au carré (x_s,i^2) modélise l'effet de saturation où la latence augmente de façon quadratique avec le nombre d'instances
    
    **Contraintes:**
    
    1. **Nombre minimal d'instances**: Pour chaque service i, ∑(x_s,i) ≥ D_i pour tous les serveurs s
    
    2. **Capacité CPU**: Pour chaque serveur s, ∑(c_i * x_s,i) ≤ C_s pour tous les services i
       où c_i est le CPU requis par instance du service i et C_s est la capacité CPU du serveur s
    
    3. **Non-négativité**: x_s,i ≥ 0 pour tout serveur s et service i
    
    **Approche de résolution:**
    
    1. L'algorithme SLSQP (Sequential Least Squares Programming) est utilisé pour résoudre ce problème d'optimisation non linéaire avec contraintes.
    
    2. La solution continue est ensuite arrondie aux nombres entiers les plus proches en préservant les contraintes:
       - Application d'abord de la fonction "floor" à toutes les valeurs
       - Ajout d'instances supplémentaires pour satisfaire les contraintes minimales
       - Priorité donnée aux serveurs avec la latence la plus faible lors de l'ajout d'instances
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
    
    2. **Optimisation**:
       - Cliquez sur le bouton "Lancer l'optimisation" pour exécuter l'algorithme
       - Analysez les résultats présentés sous forme de graphiques et de tableaux
       - Comparez les solutions continue et entière
    
    3. **Interprétation**:
       - Examinez la répartition des services entre les serveurs
       - Vérifiez l'utilisation des ressources CPU sur chaque serveur
       - Analysez les contributions à la latence totale
       - Identifiez les facteurs clés qui ont influencé la solution optimale
    
    4. **Ajustement**:
       - Modifiez les paramètres pour explorer différents scénarios
       - Observez comment les changements affectent l'allocation optimale
    """)