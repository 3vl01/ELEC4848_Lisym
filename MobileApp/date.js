const counting = (arrayTime) => {
    const arrDur = [];
    let duration = 4;

    for (let i = 0; i < arrayTime.length - 1; i++) {
        const currentTime = arrayTime[i];
        const nextTime = arrayTime[i + 1];

        const currentHour = parseInt(currentTime.substring(0, 2));
        const currentMinute = parseInt(currentTime.substring(2, 4));
        const currentSecond = parseInt(currentTime.substring(4, 6));

        const nextHour = parseInt(nextTime.substring(0, 2));
        const nextMinute = parseInt(nextTime.substring(2, 4));
        const nextSecond = parseInt(nextTime.substring(4, 6));

        const timeDifference = (nextHour - currentHour) * 3600 + (nextMinute - currentMinute) * 60 + (nextSecond - currentSecond);

        if (timeDifference <= 4) {
            duration+=timeDifference;
        } else {
            arrDur.push(duration);
            duration = 4;
        }
    }
    if (arrayTime.length == 1){
        arrDur.push(4);
    }
    else{
        arrDur.push(duration);
    }
    const count = arrDur.length;
    return {count: count, arrDur: arrDur};
}
export class date {
    constructor(obj, da) {
        this.day = da; 
        this.burning = obj.burning[da] !== undefined ? counting(Object.keys(obj.burning[da])) : {count: 0, arrDur: []}; 
        this.coughing = obj.coughing[da] !== undefined ? counting(Object.keys(obj.coughing[da])) : {count: 0, arrDur: []}; 
        this.gasStove = obj.gasStove[da] !== undefined ? counting(Object.keys(obj.gasStove[da])) : {count: 0, arrDur: []}; 
        this.glass_breaking = obj.glass_breaking[da] !== undefined ? counting(Object.keys(obj.glass_breaking[da])) : {count: 0, arrDur: []}; 
        this.plasticCollapse = obj.plasticCollapse[da] !== undefined ? counting(Object.keys(obj.plasticCollapse[da])) : {count: 0, arrDur: []}; 
        this.runningTapWater = obj.runningTapWater[da] !== undefined ? counting(Object.keys(obj.runningTapWater[da])) : {count: 0, arrDur: []}; 
        this.sneezing = obj.sneezing[da] !== undefined ? counting(Object.keys(obj.sneezing[da])) : {count: 0, arrDur: []}; 
        this.sniffingNose = obj.sniffingNose[da] !== undefined ? counting(Object.keys(obj.sniffingNose[da])) : {count: 0, arrDur: []}; 
    }       
}