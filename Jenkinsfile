pipeline {
    agent any

    stages {
        stage('Clone Repository') {
            steps {
                echo 'Cloning the repository...'
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                echo 'Setting up the Python environment...'
                sh '''
                python3 -m venv venv
                source venv/bin/activate
                pip install -r requirements.txt || echo "No requirements file found"
                '''
            }
        }

        stage('Train Model') {
            steps {
                echo 'Running training script...'
                sh '''
                source venv/bin/activate
                python train.py
                '''
            }
        }

        stage('Predict') {
            steps {
                echo 'Running prediction script...'
                sh '''
                source venv/bin/activate
                python predict.py
                '''
            }
        }
    }

    post {
        always {
            echo 'Pipeline execution complete.'
        }
    }
}
