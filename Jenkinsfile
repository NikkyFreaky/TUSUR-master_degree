pipeline {
    agent any

    stages {
        stage('Checkout') {
            when {
                branch 'main'
            }
            steps {
                checkout scm
            }
        }

        stage('Start network') {
            when {
                branch 'main'
            }
            steps {
                sh 'docker network inspect lab2-net >/dev/null 2>&1 || docker network create lab2-net'
            }
        }

        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                dir('laboratory_works/MA/lab2/service_1') {
                    sh 'docker compose -f docker-compose.yaml up -d --build'
                }
                dir('laboratory_works/MA/lab2/service_2') {
                    sh 'docker compose -f docker-compose.yaml up -d --build'
                }
                dir('laboratory_works/MA/lab2/gateway') {
                    sh 'docker compose -f docker-compose.yaml up  -d'
                }
            }
        }
    }

    post {
        always {
            sh 'docker image prune -f'
        }
        success {
            echo 'Pipeline завершен успешно'
            sh 'docker ps'
        }
        failure {
            echo 'Pipeline завершен успешно'
            sh 'docker ps'
        }
    }
}
