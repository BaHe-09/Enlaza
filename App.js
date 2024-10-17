import React from 'react';
import { StyleSheet, Text, View, Image, TouchableOpacity, ImageBackground } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import LoginScreen from './screens/LoginScreen';
import ProfileScreen from './screens/ProfileScreen';
import RegisterScreen from './screens/RegisterScreen';
import CallScreen from './screens/VideoCallScreen';
import { StatusBar } from 'expo-status-bar';
import { Ionicons, AntDesign } from '@expo/vector-icons';

// Creación del stack y tab navigator
const Stack = createNativeStackNavigator();
const Tab = createBottomTabNavigator();

const HomeScreen = ({ navigation }) => {
  const iniciarCamara = () => {
    navigation.navigate('Llamar');
  };
  const irALogin = () => {
    navigation.navigate('Login');
  };
  const irARegistrar = () => {
    navigation.navigate('Register');
  };

  return (
    <ImageBackground 
      source={require('./assets/FONDO.jpeg')} // Ruta de la imagen de fondo
      style={styles.background}
    >
      <View style={styles.container}>
        <View style={styles.header}>
          <Image source={require('./assets/Enlaza.png')} style={styles.logo} />
          <Text style={styles.headerText}>enlaza</Text>
          <TouchableOpacity style={styles.profileIcon} onPress={() => console.log('Perfil')}>
            {/* Icono de perfil aquí */}
          </TouchableOpacity>
        </View>
        <View style={styles.content}>
          <Text style={styles.welcome}>Videollamadas con LSM</Text>
          <TouchableOpacity style={styles.button} onPress={iniciarCamara}>
            <Ionicons name="videocam" size={24} color="white" />
            <Text style={styles.buttonText}>Llamar</Text>
          </TouchableOpacity>
          <TouchableOpacity onPress={irALogin}>
            <Text style={styles.linkText}>Iniciar Sesión</Text>
          </TouchableOpacity>
          <TouchableOpacity onPress={irARegistrar}>
            <Text style={styles.linkText}>Registrarse</Text>
          </TouchableOpacity>
        </View>
        <StatusBar style="auto" hidden={true} />
      </View>
    </ImageBackground>
  );
};

const ContactosScreen = () => (
  <View style={styles.container}>
    <Text>Pantalla de Contactos</Text>
  </View>
);

const SettingsScreen = () => (
  <View style={styles.container}>
    <Text>Ajustes</Text>
  </View>
);

const QaaScreen = () => (
  <View style={styles.container}>
    <Text>Preguntas y respuestas</Text>
  </View>
);

// Definición del Tab Navigator
const Tabs = () => (
  <Tab.Navigator>
    <Tab.Screen 
      name="Inicio" 
      component={HomeScreen} 
      options={{ 
        tabBarIcon: () => <Ionicons name="home" size={24} color="black" /> 
      }} 
    />
    <Tab.Screen 
      name="Contactos" 
      component={ContactosScreen} 
      options={{ 
        tabBarIcon: () => <AntDesign name="team" size={24} color="black" /> 
      }} 
    />
    <Tab.Screen 
      name="Perfil" 
      component={ProfileScreen} 
      options={{ 
        tabBarIcon: () => <AntDesign name="user" size={24} color="black" /> 
      }} 
    />
    <Tab.Screen 
      name="Ajustes" 
      component={SettingsScreen} 
      options={{ 
        tabBarIcon: () => <AntDesign name="setting" size={24} color="black" /> 
      }} 
    />
    <Tab.Screen 
      name="Dudas" 
      component={QaaScreen} 
      options={{ 
        tabBarIcon: () => <AntDesign name="question" size={24} color="black" /> 
      }} 
    />
  </Tab.Navigator>
);

// Definición del Stack Navigator
const App = () => (
  <NavigationContainer>
    <Stack.Navigator initialRouteName="Tabs">
      <Stack.Screen name="Tabs" component={Tabs} options={{ headerShown: false }} />
      <Stack.Screen name="Llamar" component={CallScreen} options={{ title: 'Llamada' }} />
      <Stack.Screen name="Login" component={LoginScreen} options={{ title: 'Iniciar Sesión' }} />
      <Stack.Screen name="Register" component={RegisterScreen} options={{ title: 'Registrarse' }} />
    </Stack.Navigator>
  </NavigationContainer>
);

const styles = StyleSheet.create({
  background: {
    flex: 1,
    resizeMode: 'cover', // Ajusta el tamaño de la imagen
    justifyContent: 'center',
  },
  container: {
    flex: 1,
    backgroundColor: 'transparent',
    padding: 20,
    justifyContent: 'center',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 40,
  },
  logo: {
    width: 50,
    height: 50,
    marginRight: 10,
  },
  headerText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  profileIcon: {
    position: 'absolute',
    right: 15,
    top: 30,
  },
  content: {
    flex: 1,
    justifyContent: 'flex-start',
    alignItems: 'center',
    marginTop: 50,
  },
  welcome: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#007bff',
    paddingVertical: 15,
    paddingHorizontal: 20,
    borderRadius: 10,
    marginBottom: 10,
    width: '80%',
    justifyContent: 'center',
  },
  buttonText: {
    color: 'white',
    fontSize: 18,
    marginLeft: 10,
  },
  linkText: {
    color: '#007BFF',
    fontSize: 16,
    marginTop: 10,
  },
  footer: {
    alignItems: 'center',
    paddingVertical: 10,
    borderTopWidth: 1,
    borderTopColor: '#ddd',
  },
});

export default App;
