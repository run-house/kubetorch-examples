#!/bin/bash

# Simple Airflow 3.0.2 installation script for EKS
set -e

echo "🚀 Installing Airflow 3.0.2 on EKS..."

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed. Please install kubectl first."
    exit 1
fi

# Check if helm is available
if ! command -v helm &> /dev/null; then
    echo "❌ helm is not installed. Please install helm first."
    exit 1
fi

# Check if we're connected to a cluster
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Not connected to a Kubernetes cluster. Please configure your kubeconfig."
    exit 1
fi

# Generate a proper fernet key
FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")

echo "🔑 Generated Fernet key for encryption"

# Update the values.yaml with the real fernet key
sed -i.bak "s/your-fernet-key-here-replace-this-with-a-real-key/$FERNET_KEY/g" values.yaml

# Add the Airflow Helm repository if not already added
if ! helm repo list | grep -q "apache-airflow"; then
    echo "📦 Adding Apache Airflow Helm repository..."
    helm repo add apache-airflow https://airflow.apache.org/charts
    helm repo update
fi

# Create namespace if it doesn't exist
NAMESPACE="airflow"
if ! kubectl get namespace $NAMESPACE &> /dev/null; then
    echo "📁 Creating namespace: $NAMESPACE"
    kubectl create namespace $NAMESPACE
fi

# Install Airflow
echo "🔧 Installing Airflow with Helm..."
helm install airflow apache-airflow/airflow \
    --namespace $NAMESPACE \
    --values values.yaml \
    --wait \
    --timeout 10m

echo "✅ Airflow installation completed!"
echo ""
echo "📋 Next steps:"
echo "1. Wait for all pods to be ready:"
echo "   kubectl get pods -n $NAMESPACE -w"
echo ""
echo "2. Get the LoadBalancer URL:"
echo "   kubectl get svc -n $NAMESPACE airflow-webserver"
echo ""
echo "3. Access Airflow UI with:"
echo "   Username: admin"
echo "   Password: admin123"
echo ""
echo "4. To uninstall:"
echo "   helm uninstall airflow -n $NAMESPACE" 