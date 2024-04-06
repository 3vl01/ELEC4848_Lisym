import moment from 'moment'; 
import React, { useEffect, useState } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import Ionicons from 'react-native-vector-icons/Ionicons';
import { initializeApp } from "firebase/app";
import { getDatabase } from 'firebase/database';
import {ref, onValue} from 'firebase/database';
import HomePage from './HomePage';
import RealtimePage from './RealtimePage';
import TrendsPage from './TrendsPage';
import {date} from './date';

const Tab = createBottomTabNavigator();

const firebaseConfig = {
  apiKey: "AIzaSyCD-KyzRERIwah1_BCFbltVQOUdtgLbc3U",
  authDomain: "fypfirebase-389cb.firebaseapp.com",
  databaseURL: "https://fypfirebase-389cb-default-rtdb.firebaseio.com",
  projectId: "fypfirebase-389cb",
  storageBucket: "fypfirebase-389cb.appspot.com",
  messagingSenderId: "151449890420",
  appId: "1:151449890420:web:ccd6a8776775042e092b21",
  measurementId: "G-FNTQ6GXC0J"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
// const analytics = getAnalytics(app);
const db = getDatabase(app);

const App = () => {
  const [data, setData] = useState(new Map());
  var arrayByDates = [];
  useEffect(() => {
    const fetchData = () => {
        onValue(ref(db, '/Tryfirebase/class'), (snapshot) => {
            const newData = snapshot.val() || {};
            setData(newData);
        });
    };

    // Fetch data initially
    fetchData();

    // Fetch data every second
    const interval = setInterval(fetchData, 5000);

    // Clean up the interval when the component unmounts
    return () => clearInterval(interval);
  }, []);

  const secondLevelChildNames = new Set();
  
  Object.values(data).forEach((firstLevelChild) => {
      Object.keys(firstLevelChild).forEach((secondLevelChildName) => {
      secondLevelChildNames.add(secondLevelChildName);
      });
  });
  const secondLevelChildNamesArray = Array.from(secondLevelChildNames).sort();

  const today = moment().format("YYMMDD").toString();
  for (let day of secondLevelChildNamesArray){
    if(day != today){
      const da = new date(data, day);
      arrayByDates.push(da);
    }
  }

  if (arrayByDates.length != 0){
    const todayData = new date(data, today);
    return (
      <NavigationContainer>
        <Tab.Navigator screenOptions={({ route }) => ({
            tabBarIcon: ({ focused, color, size }) => {
              let iconName;
  
              if (route.name === 'Home') {
                iconName = focused ? 'home' : 'home-outline';
              } else if (route.name === 'Realtime') {
                iconName = focused ? 'radio' : 'radio-outline';
              } else if (route.name === 'Trends') {
                iconName = focused ? 'stats-chart' : 'stats-chart-outline';
              }
  
              // You can return any component that you like here!
              return <Ionicons name={iconName} size={size} color={color} />;
            },
            tabBarActiveTintColor: 'tomato',
            tabBarInactiveTintColor: 'gray',
          })}>
          <Tab.Screen name="Home" children={() => <HomePage tdata={todayData}/>} />
          <Tab.Screen name="Realtime" children={() => <RealtimePage tdata={todayData}/>} />
          <Tab.Screen name="Trends" children={() => <TrendsPage data={data}/>} />
        </Tab.Navigator>
      </NavigationContainer>
    );
  }
};

export default App;