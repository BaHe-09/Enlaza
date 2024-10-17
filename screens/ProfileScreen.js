// ProfileScreen.js
import React from 'react';
import { StyleSheet, View, Text, Image } from 'react-native';

const ProfileScreen = () => {
  // Aquí puedes definir los datos del usuario. Por ahora, usaré valores estáticos.
  const userData = {
    nombre: 'Juan Pérez',
    correo: 'juan.perez@example.com',
    telefono: '123-456-7890',
    fotoPerfil: require('../assets/miau.png'), // Ruta corregida a la imagen de perfil
  };

  return (
    <View style={styles.container}>
      <Image source={userData.fotoPerfil} style={styles.profileImage} />
      <Text style={styles.name}>{userData.nombre}</Text>
      <Text style={styles.info}>Correo: {userData.correo}</Text>
      <Text style={styles.info}>Teléfono: {userData.telefono}</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    padding: 20,
  },
  profileImage: {
    width: 100,
    height: 100,
    borderRadius: 50, // Hace que la imagen sea circular
    marginBottom: 20,
  },
  name: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  info: {
    fontSize: 16,
    color: '#333',
    marginBottom: 5,
  },
});

export default ProfileScreen;