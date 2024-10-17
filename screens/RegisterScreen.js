// RegisterScreen.js
import React, { useState } from 'react';
import { View, Text, TextInput, Button, StyleSheet, Image } from 'react-native';

const RegisterScreen = ({ navigation }) => {
  const [nombre, setNombre] = useState('');
  const [correo, setCorreo] = useState('');
  const [telefono, setTelefono] = useState('');
  const [contraseña, setContraseña] = useState('');

  const handleRegister = () => {
    // Aquí iría la lógica para registrar al usuario
    console.log('Nombre:', nombre);
    console.log('Correo:', correo);
    console.log('Teléfono:', telefono);
    console.log('Contraseña:', contraseña);
    navigation.navigate('Login'); // Navegar a la pantalla de inicio de sesión después de registrarse
  };

  return (
    <View style={styles.container}>
      {/* Agregar el logo de LSM y el mensaje de bienvenida */}
      <Image source={require('../assets/Enlaza.png')} style={styles.logo} />
      <Text style={styles.welcomeText}>¡Bienvenido! Crea tu cuenta</Text>
      
      <Text style={styles.label}>Nombre</Text>
      <TextInput
        style={styles.input}
        value={nombre}
        onChangeText={setNombre}
        placeholder="Nombre completo"
      />

      <Text style={styles.label}>Correo</Text>
      <TextInput
        style={styles.input}
        value={correo}
        onChangeText={setCorreo}
        placeholder="Correo electrónico"
        keyboardType="email-address"
      />

      <Text style={styles.label}>Teléfono</Text>
      <TextInput
        style={styles.input}
        value={telefono}
        onChangeText={setTelefono}
        placeholder="Número de teléfono"
        keyboardType="phone-pad"
      />

      <Text style={styles.label}>Contraseña</Text>
      <TextInput
        style={styles.input}
        value={contraseña}
        onChangeText={setContraseña}
        placeholder="Contraseña"
        secureTextEntry
      />

      <Button title="Registrarse" onPress={handleRegister} />

      <Text style={styles.loginText}>
        ¿Ya tienes una cuenta?{' '}
        <Text style={styles.loginLink} onPress={() => navigation.navigate('Login')}>
          Inicia sesión
        </Text>
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
    justifyContent: 'center',
    paddingHorizontal: 20,
  },
  logo: {
    width: 100,
    height: 100,
    alignSelf: 'center',
    marginBottom: 20,
  },
  welcomeText: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
    color: '#007BFF',
  },
  label: {
    fontSize: 16,
    marginBottom: 5,
    color: '#333',
  },
  input: {
    borderWidth: 1,
    borderColor: '#CCCCCC',
    padding: 10,
    marginBottom: 15,
    borderRadius: 5,
  },
  loginText: {
    marginTop: 20,
    textAlign: 'center',
    fontSize: 16,
  },
  loginLink: {
    color: '#007BFF',
    fontWeight: 'bold',
  },
});

export default RegisterScreen;