// screens/CallScreen.js
import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Ionicons, MaterialIcons, Entypo } from '@expo/vector-icons';

const CallScreen = () => {
  return (
    <View style={styles.container}>
      <TouchableOpacity style={styles.menuIcon}>
        <Entypo name="menu" size={24} color="white" />
      </TouchableOpacity>

      <View style={styles.centerContent}>
        <View style={styles.avatar} />
        <Text style={styles.username}>Username</Text>
        <Text style={styles.timer}>00:12</Text>
      </View>

      <View style={styles.controls}>
        <TouchableOpacity style={styles.buttonEndCall}>
          <Ionicons name="call" size={28} color="white" />
        </TouchableOpacity>
        <TouchableOpacity style={styles.buttonControl}>
          <MaterialIcons name="mic" size={28} color="white" />
        </TouchableOpacity>
        <TouchableOpacity style={styles.buttonControl}>
          <Ionicons name="videocam" size={28} color="white" />
        </TouchableOpacity>
        <TouchableOpacity style={styles.buttonControl}>
          <Ionicons name="chatbubble" size={28} color="white" />
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#007BFF',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 20,
  },
  menuIcon: {
    position: 'absolute',
    top: 40,
    left: 20,
  },
  centerContent: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  avatar: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    marginBottom: 10,
  },
  username: {
    color: 'white',
    fontSize: 20,
    fontWeight: 'bold',
  },
  timer: {
    color: 'white',
    fontSize: 16,
    marginTop: 5,
  },
  controls: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    width: '100%',
    paddingHorizontal: 30,
    marginBottom: 30,
  },
  buttonEndCall: {
    backgroundColor: 'red',
    padding: 15,
    borderRadius: 50,
  },
  buttonControl: {
    backgroundColor: '#1A1A1A',
    padding: 15,
    borderRadius: 50,
  },
});

export default CallScreen;