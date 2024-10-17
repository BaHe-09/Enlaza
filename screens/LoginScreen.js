// LoginScreen.js
import React, { useState } from 'react';
import { View, Text, TextInput, Button, StyleSheet, Image, TouchableOpacity } from 'react-native';

const LoginScreen = ({ navigation }) => {
  const [correo, setCorreo] = useState('');
  const [contraseña, setContraseña] = useState('');

  const handleLogin = () => {
    // Aquí iría la lógica para iniciar sesión
    console.log('Correo:', correo);
    console.log('Contraseña:', contraseña);
    // Navegar a la pantalla principal o a la pantalla de perfil después de iniciar sesión
    navigation.navigate('Profile');
  };

  const handleForgotPassword = () => {
    // Lógica para manejar el caso de "Olvidaste tu contraseña o correo"
    console.log('Redirigir a la pantalla de recuperación de contraseña o asistencia.');
    // Aquí puedes navegar a una pantalla de recuperación si la tienes implementada
  };

  return (
    <View style={styles.container}>
      <Image source={require('../assets/Enlaza.png')} style={styles.logo} />
      <Text style={styles.welcomeText}>Inicia Sesión</Text>

      <Text style={styles.label}>Correo</Text>
      <TextInput
        style={styles.input}
        value={correo}
        onChangeText={setCorreo}
        placeholder="Correo electrónico"
        keyboardType="email-address"
      />

      <Text style={styles.label}>Contraseña</Text>
      <TextInput
        style={styles.input}
        value={contraseña}
        onChangeText={setContraseña}
        placeholder="Contraseña"
        secureTextEntry
      />

      <Button title="Iniciar Sesión" onPress={handleLogin} />

      {/* Texto "Olvidaste tu contraseña o correo" */}
      <TouchableOpacity onPress={handleForgotPassword}>
        <Text style={styles.forgotPasswordText}>¿Olvidaste tu contraseña o correo?</Text>
      </TouchableOpacity>

      <Text style={styles.registerText}>
        ¿Eres nuevo?{' '}
        <Text style={styles.registerLink} onPress={() => navigation.navigate('Register')}>
          Regístrate
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
  forgotPasswordText: {
    color: '#007BFF',
    fontSize: 16,
    textAlign: 'center',
    marginVertical: 10,
    fontWeight: 'bold',
  },
  registerText: {
    marginTop: 20,
    textAlign: 'center',
    fontSize: 16,
  },
  registerLink: {
    color: '#007BFF',
    fontWeight: 'bold',
  },
});

export default LoginScreen;