# Kubetorch + Airflow

This is a standalone example that demonstrates a simple use of Kubetorch inside an Airflow DAG.

## Overview

TK - talk about the usage flow. What does Kubetorch unlock?

TK - talk about the components.

## Installation & Setup

Ideally, you should be running Kubetorch and Airflow in the same Kubernetes cluster. This will simplify connections between services and improve overall reliability.

To illustrate how to set up Airflow + Kubetorch on a fresh cluster, we'll walk through deploying Airflow with Helm, using a Docker image with Airflow Kubetorch Python package and our training DAG.

You will need a Kubernetes cluster, a local Kubeconfig, and 

1. Replace `<KT_VERSION>`, `KT_LICENSE_KEY`, and `<API_KEY>` in the URL in `requirements.txt`.
   You can also directly use the URL provided in our [Onboarding Guide](https://www.run.house/kubetorch/get-started)
2. Build the Docker image, tagging an appropriate location in your registry (e.g. ECR, GCR). This should be accessible from your Kubernetes cluster.
   ```bash
   docker build --platform linux/amd64 -t your-registry/kt-airflow-custom:latest .
   ```
3. Push the image to your container registry.
   ```bash
   docker push your-registry/kt-airflow-custom:latest
   ```
4. Use Helm to install Airflow with our custom image.
   ```bash
   helm repo add apache-airflow https://airflow.apache.org
   helm repo update
   kubectl create namespace airflow
   helm install airflow apache-airflow/airflow \
     --namespace airflow \
     --set images.airflow.repository=your-registry/kt-airflow-custom \
     --set images.airflow.tag=latest
   ```
   Depending on your cluster setup, may need to use a custom `values.yaml` file for the installation. The one in this folder works with the Kubetorch Terraform install on EKS.
5. Apply RBAC to the Airflow worker service account to allow for Kubetorch dispatching operations.
   ```bash
   kubectl apply -f rbac.yaml
   ```

### Connecting to Airflow

```bash
kubectl port-forward svc/airflow-api-server 8080:8080 --namespace airflow
```

Then navigate to http://localhost:8080 in your browser to view the Airflow UI.

By default, you should be able to log in with username `admin` and password `admin`.

### Monitoring and Debugging

The following Kubernetes commands will be useful for debugging:

```bash
# List all airflow pods
kubectl get pods -n [your-namespace] | grep airflow

# View logs for specific components
kubectl logs -f deployment/airflow-scheduler -n [your-namespace]
kubectl logs -f deployment/airflow-webserver -n [your-namespace]
kubectl logs -f deployment/airflow-worker -n [your-namespace]
```

To see logs from the Kubetorch methods, launching compute and more, you'll likely be most interested in `scheduler`.

## Additional Information

### Airflow Installation

Installation on a fresh Kubernetes cluster:

```bash
# Optional, ensure EBS is installed on your cluster for PVCs to bind properly
$ kubectl apply -k "github.com/kubernetes-sigs/aws-ebs-csi-driver/deploy/kubernetes/overlays/stable/?ref=release-1.28"
# Install Airflow
$ helm upgrade --install airflow apache-airflow/airflow --namespace airflow  --create-namespace -f values-simple.yaml
# Confirm that the Airflow pods are Ready
$ kubectl get pods -n airflow
# Install Kubetorch on the cluster
$ helm upgrade --install kubetorch <path_to_kubetorch> -n kubetorch --create-namespace
# Open the UI Dashboard on localhost:8080
$ kubectl port-forward svc/airflow-api-server 8080:8080 --namespace airflow
```

Then, navigate to `localhost:8080` in your browser and use the default username `admin` and password `admin` to autheticate. You should see the DAGs included in your image in the interface.