CREATE TABLE students (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(100) NOT NULL,
  group_number VARCHAR(10) NOT NULL
);

INSERT INTO students (name, group_number) VALUES
('Ivan Ivanov', '425-M'),
('Petr Petrov', '425-M'),
('Semen Semenov', '425-M');
